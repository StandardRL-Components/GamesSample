
# Generated: 2025-08-28T03:16:55.907339
# Source Brief: brief_04872.md
# Brief Index: 4872

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Constants
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 400
TILE_SIZE = 40
GRID_WIDTH = SCREEN_WIDTH // TILE_SIZE  # 16
GRID_HEIGHT = SCREEN_HEIGHT // TILE_SIZE # 10
NUM_BOXES = 5
MAX_STEPS = 6000 # As per brief
LEVEL_GEN_PULLS = 40 # Number of random moves to scramble the level

# Colors
COLOR_BG = (25, 25, 35)
COLOR_GRID = (40, 40, 50)
COLOR_WALL = (80, 80, 90)
COLOR_GOAL = (40, 100, 60)
COLOR_GOAL_FILLED = (80, 160, 100)
COLOR_BOX = (160, 100, 40)
COLOR_BOX_SHADOW = (110, 70, 30)
COLOR_PLAYER = (220, 50, 50)
COLOR_PLAYER_SHADOW = (150, 35, 35)
COLOR_TEXT = (220, 220, 220)
COLOR_PARTICLE = (255, 255, 150)

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, dx, dy, life):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.life = life
        self.max_life = life

    def update(self):
        self.x += self.dx
        self.y += self.dy
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            # Fade out and shrink
            alpha = int(255 * (self.life / self.max_life))
            radius = int(3 * (self.life / self.max_life))
            if radius > 0:
                # Create a temporary surface for alpha blending
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, COLOR_PARTICLE + (alpha,), (radius, radius), radius)
                surface.blit(temp_surf, (int(self.x - radius), int(self.y - radius)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Push boxes onto the green goals."
    )

    game_description = (
        "Push boxes into their designated goals before the step limit runs out in this top-down puzzle game."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 18)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        self.player_pos = [0, 0]
        self.box_positions = []
        self.goal_positions = []
        self.wall_positions = set()
        self.particles = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""

        self.reset()
        
        self.validate_implementation()

    def _generate_level(self):
        self.wall_positions.clear()
        self.goal_positions.clear()
        self.box_positions.clear()
        self.particles.clear()

        # Create perimeter walls
        for x in range(GRID_WIDTH):
            self.wall_positions.add((x, 0))
            self.wall_positions.add((x, GRID_HEIGHT - 1))
        for y in range(GRID_HEIGHT):
            self.wall_positions.add((0, y))
            self.wall_positions.add((GRID_WIDTH - 1, y))

        # Generate potential spawn points (not walls)
        possible_spawns = []
        for x in range(1, GRID_WIDTH - 1):
            for y in range(1, GRID_HEIGHT - 1):
                possible_spawns.append((x, y))
        
        # Use np_random for reproducibility
        spawn_indices = self.np_random.choice(len(possible_spawns), size=NUM_BOXES, replace=False)
        self.goal_positions = [possible_spawns[i] for i in spawn_indices]
        
        # Start with boxes on goals (guarantees solvability)
        self.box_positions = [list(pos) for pos in self.goal_positions]

        # "Pull" boxes randomly to scramble the puzzle
        for _ in range(LEVEL_GEN_PULLS):
            box_idx = self.np_random.integers(0, NUM_BOXES)
            box_pos = self.box_positions[box_idx]
            
            # Try a random pull direction
            move_dir = self.np_random.integers(0, 4)
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][move_dir]

            # A "pull" is the inverse of a push
            new_box_pos = (box_pos[0] - dx, box_pos[1] - dy)
            required_player_pos = (box_pos[0] + dx, box_pos[1] + dy)
            
            # Check if the move is valid
            is_valid = True
            if new_box_pos in self.wall_positions: is_valid = False
            if new_box_pos in [tuple(p) for p in self.box_positions]: is_valid = False
            if required_player_pos in self.wall_positions: is_valid = False
            
            if is_valid:
                self.box_positions[box_idx] = list(new_box_pos)

        # Place player in a random empty spot
        occupied = self.wall_positions.union(set(map(tuple, self.box_positions)))
        empty_spots = [p for p in possible_spawns if p not in occupied]
        if not empty_spots: # Fallback if everything is somehow full
            self.reset() # This is unlikely but a good safeguard
            return
            
        player_idx = self.np_random.choice(len(empty_spots))
        self.player_pos = list(empty_spots[player_idx])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._generate_level()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        old_box_positions = [tuple(p) for p in self.box_positions]
        
        # --- Action Logic ---
        if movement != 0: # 0 is no-op
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            
            player_next_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)

            # Check for wall collision
            if player_next_pos in self.wall_positions:
                pass # Player hits a wall
            
            # Check for box collision (push)
            elif player_next_pos in old_box_positions:
                box_idx = old_box_positions.index(player_next_pos)
                box_next_pos = (player_next_pos[0] + dx, player_next_pos[1] + dy)
                
                # Check if box can be pushed
                if box_next_pos not in self.wall_positions and box_next_pos not in old_box_positions:
                    # Successful push
                    self.box_positions[box_idx] = list(box_next_pos)
                    self.player_pos = list(player_next_pos)
                    # sfx: push_box.wav
                    # Spawn particles
                    px, py = (player_next_pos[0] + 0.5) * TILE_SIZE, (player_next_pos[1] + 0.5) * TILE_SIZE
                    for _ in range(5):
                        p_dx = (self.np_random.random() - 0.5) * 2
                        p_dy = (self.np_random.random() - 0.5) * 2
                        self.particles.append(Particle(px, py, p_dx, p_dy, life=15))
            
            # Empty space movement
            else:
                self.player_pos = list(player_next_pos)
        
        # --- Reward Calculation ---
        moved_box_info = self._get_moved_box_info(old_box_positions)
        if moved_box_info:
            box_old_pos, box_new_pos = moved_box_info
            
            # Distance change reward
            old_dist = min(self._manhattan_distance(box_old_pos, g) for g in self.goal_positions)
            new_dist = min(self._manhattan_distance(box_new_pos, g) for g in self.goal_positions)
            
            if new_dist < old_dist:
                reward += 0.1 # Moved closer to a goal
            else:
                reward -= 0.01 # Moved away or parallel
            
            # Box on goal reward
            if tuple(box_new_pos) in self.goal_positions and box_old_pos not in self.goal_positions:
                reward += 1.0 # sfx: goal_achieved.wav
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            self.game_over = True
            if self._check_win_condition():
                reward += 100 # Win bonus
                self.win_message = "SUCCESS!"
                # sfx: win_jingle.wav
            else:
                reward -= 10 # Timeout penalty
                self.win_message = "TIME UP"
                # sfx: lose_sound.wav

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_moved_box_info(self, old_box_positions):
        current_box_set = set(map(tuple, self.box_positions))
        old_box_set = set(old_box_positions)
        
        moved_from = old_box_set - current_box_set
        moved_to = current_box_set - old_box_set
        
        if moved_from and moved_to:
            return moved_from.pop(), moved_to.pop()
        return None

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _check_win_condition(self):
        return set(map(tuple, self.box_positions)) == set(self.goal_positions)

    def _check_termination(self):
        if self._check_win_condition():
            return True
        if self.steps >= MAX_STEPS:
            return True
        return False

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _get_observation(self):
        self.screen.fill(COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, SCREEN_WIDTH, TILE_SIZE):
            pygame.draw.line(self.screen, COLOR_GRID, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, TILE_SIZE):
            pygame.draw.line(self.screen, COLOR_GRID, (0, y), (SCREEN_WIDTH, y))

        # Draw goals
        for gx, gy in self.goal_positions:
            is_filled = [gx, gy] in self.box_positions
            color = COLOR_GOAL_FILLED if is_filled else COLOR_GOAL
            pygame.draw.rect(self.screen, color, (gx * TILE_SIZE, gy * TILE_SIZE, TILE_SIZE, TILE_SIZE))

        # Draw walls
        for wx, wy in self.wall_positions:
            pygame.draw.rect(self.screen, COLOR_WALL, (wx * TILE_SIZE, wy * TILE_SIZE, TILE_SIZE, TILE_SIZE))

        # Draw boxes
        for bx, by in self.box_positions:
            px, py = bx * TILE_SIZE, by * TILE_SIZE
            shadow_offset = TILE_SIZE * 0.1
            pygame.draw.rect(self.screen, COLOR_BOX_SHADOW, (px + shadow_offset, py + shadow_offset, TILE_SIZE, TILE_SIZE))
            pygame.draw.rect(self.screen, COLOR_BOX, (px, py, TILE_SIZE, TILE_SIZE))
            pygame.draw.rect(self.screen, COLOR_BG, (px, py, TILE_SIZE, TILE_SIZE), 2) # Border

        # Draw player
        px, py = self.player_pos[0] * TILE_SIZE, self.player_pos[1] * TILE_SIZE
        shadow_offset = TILE_SIZE * 0.1
        pygame.draw.rect(self.screen, COLOR_PLAYER_SHADOW, (px + shadow_offset, py + shadow_offset, TILE_SIZE, TILE_SIZE))
        pygame.draw.rect(self.screen, COLOR_PLAYER, (px, py, TILE_SIZE, TILE_SIZE))
        
        # Update and draw particles
        for p in self.particles:
            p.update()
            p.draw(self.screen)
        self.particles = [p for p in self.particles if p.life > 0]

    def _render_ui(self):
        # Score (boxes on goals)
        boxes_on_goals = len(set(map(tuple, self.box_positions)).intersection(set(self.goal_positions)))
        score_text = self.font_small.render(f"Goals: {boxes_on_goals} / {NUM_BOXES}", True, COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Time remaining (as steps)
        time_left = max(0, MAX_STEPS - self.steps)
        time_text = self.font_small.render(f"Steps Left: {time_left}", True, COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(SCREEN_WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)
        
        # Game over message
        if self.game_over:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.win_message, True, COLOR_TEXT)
            end_rect = end_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a display for manual testing
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Sokoban Puzzle")
    clock = pygame.time.Clock()

    total_reward = 0
    total_steps = 0

    action = env.action_space.sample()
    action[0] = 0 # Start with no-op

    while not done:
        # --- Human Controls ---
        movement_action = 0 # Default to no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement_action = 1
                elif event.key == pygame.K_DOWN:
                    movement_action = 2
                elif event.key == pygame.K_LEFT:
                    movement_action = 3
                elif event.key == pygame.K_RIGHT:
                    movement_action = 4
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    total_reward = 0
                    total_steps = 0
                    print("--- ENV RESET ---")
                elif event.key == pygame.K_q:
                    done = True

        # Only step if a move was made
        if movement_action != 0:
            action[0] = movement_action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_steps += 1
            print(f"Step: {total_steps}, Action: {action[0]}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}")
            if terminated:
                print("Game Over!")
        
        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate for human play

    env.close()