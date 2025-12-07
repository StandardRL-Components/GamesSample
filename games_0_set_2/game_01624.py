
# Generated: 2025-08-27T17:43:54.098301
# Source Brief: brief_01624.md
# Brief Index: 1624

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the selected robot. Press Space to cycle which robot is selected. Rescue all three robots before you run out of moves!"
    )

    game_description = (
        "A turn-based puzzle game. Strategically guide three robots to their matching exits, navigating around obstacles. Each move counts, so plan your path carefully to succeed!"
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 16
    GRID_HEIGHT = 10
    CELL_SIZE = 40
    
    COLOR_BG = (15, 15, 20)
    COLOR_GRID = (40, 40, 50)
    COLOR_OBSTACLE = (80, 80, 90)
    
    ROBOT_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
    ]
    EXIT_COLORS = [
        (120, 40, 40),   # Darker Red
        (40, 120, 40),   # Darker Green
        (40, 70, 120),   # Darker Blue
    ]
    
    COLOR_SELECTION = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_FAIL = (255, 0, 0)
    
    MAX_MOVES = 100
    NUM_OBSTACLES = 15
    NUM_ROBOTS = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 36)
        
        self.robots = []
        self.exits = []
        self.obstacles = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = 0
        self.selected_robot_idx = 0
        self.last_space_held = False
        self.move_feedback = {"robot_idx": -1, "type": "none", "timer": 0}

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = self.MAX_MOVES
        self.selected_robot_idx = 0
        self.last_space_held = False
        self.particles = []
        
        self._generate_level()
        self._find_next_selectable_robot()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_pressed = action[1] == 1
        
        reward = 0
        self.steps += 1
        
        # --- Action Handling ---
        
        # Handle selection change (Spacebar)
        # This does not consume a move
        if space_pressed and not self.last_space_held:
            self.selected_robot_idx = (self.selected_robot_idx + 1) % self.NUM_ROBOTS
            self._find_next_selectable_robot()
        self.last_space_held = space_pressed

        # Handle movement (Arrow keys)
        # This consumes a move
        if movement != 0:
            self.moves_remaining -= 1
            reward -= 0.1 # Penalty for each move
            
            robot = self.robots[self.selected_robot_idx]
            if not robot["rescued"]:
                current_pos = robot["pos"]
                
                # 0=none, 1=up, 2=down, 3=left, 4=right
                dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
                target_pos = [current_pos[0] + dx, current_pos[1] + dy]
                
                if self._is_valid_move(target_pos, self.selected_robot_idx):
                    robot["pos"] = target_pos
                    # sfx: move_success.wav
                    
                    # Check for rescue
                    if robot["pos"] == self.exits[robot["id"]]["pos"]:
                        robot["rescued"] = True
                        reward += 10
                        self._create_particles(robot["pos"], robot["color"])
                        self._find_next_selectable_robot()
                        # sfx: rescue.wav
                else:
                    self.move_feedback = {"robot_idx": self.selected_robot_idx, "type": "fail", "timer": 10}
                    # sfx: move_fail.wav
        
        # --- Update Game State & Check Termination ---
        self.score += reward
        
        rescued_count = sum(1 for r in self.robots if r["rescued"])
        
        terminated = False
        if rescued_count == self.NUM_ROBOTS:
            reward += 50 # Bonus for rescuing all robots
            self.score += 50
            terminated = True
            self.game_over_message = "ALL ROBOTS RESCUED!"
        elif self.moves_remaining <= 0:
            reward -= 100 # Penalty for running out of moves
            self.score -= 100
            terminated = True
            self.game_over_message = "OUT OF MOVES"
            
        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "robots_rescued": sum(1 for r in self.robots if r["rescued"])
        }

    # --- Rendering Methods ---

    def _render_game(self):
        self._draw_grid()
        
        for exit_data in self.exits:
            self._draw_cell(exit_data["pos"], exit_data["color"])
            
        for obstacle_pos in self.obstacles:
            self._draw_cell(obstacle_pos, self.COLOR_OBSTACLE)
            
        self._update_and_draw_particles()
        
        for i, robot in enumerate(self.robots):
            if not robot["rescued"]:
                self._draw_robot(robot, is_selected=(i == self.selected_robot_idx and not self.game_over))
    
    def _render_ui(self):
        # Moves remaining
        moves_text = self.font_medium.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 10))
        
        # Score
        score_text = self.font_medium.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.game_over_message, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _draw_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _draw_cell(self, grid_pos, color, padding=4):
        x, y = grid_pos
        rect = pygame.Rect(
            x * self.CELL_SIZE + padding,
            y * self.CELL_SIZE + padding,
            self.CELL_SIZE - 2 * padding,
            self.CELL_SIZE - 2 * padding
        )
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        
    def _draw_robot(self, robot, is_selected):
        x, y = robot["pos"]
        center_x = int(x * self.CELL_SIZE + self.CELL_SIZE / 2)
        center_y = int(y * self.CELL_SIZE + self.CELL_SIZE / 2)
        radius = int(self.CELL_SIZE / 2 * 0.7)
        
        # Selection indicator
        if is_selected:
            pulse = (math.sin(self.steps * 0.3) + 1) / 2 # 0 to 1
            sel_radius = radius + 4 + pulse * 4
            alpha = 100 + pulse * 100
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, int(sel_radius), (*self.COLOR_SELECTION, int(alpha)))
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, int(sel_radius)+1, (*self.COLOR_SELECTION, int(alpha/2)))

        # Move fail indicator
        if self.move_feedback["timer"] > 0 and self.move_feedback["robot_idx"] == robot["id"]:
            fail_radius = radius + 6
            alpha = (self.move_feedback["timer"] / 10) * 255
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, fail_radius, (*self.COLOR_FAIL, int(alpha)))
            self.move_feedback["timer"] -= 1

        # Robot body
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, robot["color"])
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, robot["color"])
        
        # Eye
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y - 3, 3, (255, 255, 255))
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y - 3, 1, (0, 0, 0))

    # --- Helper Methods ---

    def _generate_level(self):
        max_attempts = 100
        for _ in range(max_attempts):
            all_pos = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
            random.shuffle(all_pos)
            
            self.robots = []
            self.exits = []
            self.obstacles = []
            
            temp_occupied = set()
            
            # Place robots
            for i in range(self.NUM_ROBOTS):
                pos = all_pos.pop()
                self.robots.append({"pos": list(pos), "color": self.ROBOT_COLORS[i], "id": i, "rescued": False})
                temp_occupied.add(pos)
            
            # Place exits
            for i in range(self.NUM_ROBOTS):
                pos = all_pos.pop()
                self.exits.append({"pos": list(pos), "color": self.EXIT_COLORS[i], "id": i})
                temp_occupied.add(pos)
                
            # Place obstacles
            for _ in range(self.NUM_OBSTACLES):
                pos = all_pos.pop()
                self.obstacles.append(list(pos))
            
            # Validate paths
            if self._is_level_solvable():
                return
        
        raise RuntimeError("Failed to generate a solvable level after multiple attempts.")

    def _is_level_solvable(self):
        for i in range(self.NUM_ROBOTS):
            start = tuple(self.robots[i]["pos"])
            end = tuple(self.exits[i]["pos"])
            
            # Obstacles for this path include actual obstacles and other robots' start positions
            obstacle_set = {tuple(o) for o in self.obstacles}
            for j in range(self.NUM_ROBOTS):
                if i != j:
                    obstacle_set.add(tuple(self.robots[j]["pos"]))
                    
            if not self._find_path_bfs(start, end, obstacle_set):
                return False
        return True

    def _find_path_bfs(self, start, end, obstacles):
        q = deque([start])
        visited = {start}
        while q:
            x, y = q.popleft()
            if (x, y) == end:
                return True
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and
                        (nx, ny) not in obstacles and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return False

    def _is_valid_move(self, pos, moving_robot_id):
        x, y = pos
        # Check boundaries
        if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
            return False
        # Check obstacles
        if pos in self.obstacles:
            return False
        # Check other robots
        for i, robot in enumerate(self.robots):
            if i != moving_robot_id and not robot["rescued"] and robot["pos"] == pos:
                return False
        return True

    def _find_next_selectable_robot(self):
        initial_idx = self.selected_robot_idx
        for _ in range(self.NUM_ROBOTS):
            if not self.robots[self.selected_robot_idx]["rescued"]:
                return
            self.selected_robot_idx = (self.selected_robot_idx + 1) % self.NUM_ROBOTS
        # If all are rescued, it doesn't matter who is selected
        self.selected_robot_idx = initial_idx

    def _create_particles(self, grid_pos, color):
        center_x = grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2
        center_y = grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                "x": center_x,
                "y": center_y,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "life": random.randint(15, 30),
                "color": color
            })

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["life"] -= 1
            if p["life"] > 0:
                alpha = int(255 * (p["life"] / 30))
                radius = int(p["life"] / 5)
                pygame.gfxdraw.filled_circle(self.screen, int(p["x"]), int(p["y"]), max(0, radius), (*p["color"], alpha))
                active_particles.append(p)
        self.particles = active_particles

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Rescue Bots")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    terminated = False
    
    print("\n" + "="*30)
    print("      MANUAL PLAY MODE")
    print("="*30)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        movement = 0 # no-op
        space = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                
                # Map keys to actions
                if not terminated:
                    if event.key == pygame.K_UP:
                        movement = 1
                    elif event.key == pygame.K_DOWN:
                        movement = 2
                    elif event.key == pygame.K_LEFT:
                        movement = 3
                    elif event.key == pygame.K_RIGHT:
                        movement = 4
                    elif event.key == pygame.K_SPACE:
                        space = 1

        if not terminated:
            action = [movement, space, 0] # Movement, Space, Shift
            obs, reward, terminated, truncated, info = env.step(action)
            
            if movement != 0:
                print(f"Move taken. Reward: {reward:.1f}, Score: {info['score']:.1f}, Moves Left: {info['moves_remaining']}")
            if terminated:
                 print(f"\n--- GAME OVER ---")
                 print(f"Final Score: {info['score']:.1f}")
                 print(f"Press 'R' to restart.")


        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit frame rate

    pygame.quit()