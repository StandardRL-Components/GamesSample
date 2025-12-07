
# Generated: 2025-08-27T21:45:54.632622
# Source Brief: brief_02900.md
# Brief Index: 2900

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the selected robot. "
        "Pressing no key (no-op) switches to the next available robot."
    )

    game_description = (
        "A top-down puzzle game where you must guide a team of three robots to a rescue zone, "
        "avoiding deadly pits and moving obstacles. Plan your moves carefully!"
    )

    auto_advance = False

    # --- Game Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CELL_SIZE = 40
    GRID_WIDTH = SCREEN_WIDTH // CELL_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // CELL_SIZE
    MAX_STEPS = 500

    # --- Colors ---
    COLOR_BG = (25, 35, 45)
    COLOR_GRID = (40, 50, 60)
    COLOR_RESCUE_ZONE = (40, 180, 120, 100)
    COLOR_PIT = (0, 0, 0)
    COLOR_OBSTACLE = (100, 110, 120)
    COLOR_ROBOTS = [(255, 80, 80), (80, 150, 255), (255, 255, 100)]
    COLOR_TEXT = (220, 220, 220)
    COLOR_SELECTION = (255, 255, 255)

    def __init__(self, render_mode="rgb_array", level=1):
        super().__init__()

        self.render_mode = render_mode
        self.initial_level = level

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)

        self.robots = []
        self.pits = []
        self.obstacles = []
        self.particles = []
        self.rescue_zone = None
        self.active_robot_indices = []
        self.active_robot_selector = 0
        self.selected_robot_idx = 0
        
        self.steps = 0
        self.score = 0
        self.level = self.initial_level
        self.game_over_message = ""
        self.game_over = False

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and 'level' in options:
            self.level = options['level']
        else:
            self.level = self.initial_level

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.particles.clear()
        
        self._generate_level()
        self._update_active_robots()
        
        if not self.active_robot_indices: # Handle case where level generation fails
             self._generate_level()
             self._update_active_robots()
             
        self.selected_robot_idx = self.active_robot_indices[0] if self.active_robot_indices else 0


        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.robots.clear()
        self.pits.clear()
        self.obstacles.clear()

        # Define rescue zone on the right
        rescue_width = 3
        self.rescue_zone = pygame.Rect(
            (self.GRID_WIDTH - rescue_width) * self.CELL_SIZE, 0,
            rescue_width * self.CELL_SIZE, self.SCREEN_HEIGHT
        )
        self.rescue_zone_center = (self.GRID_WIDTH - rescue_width / 2, self.GRID_HEIGHT / 2)

        # Generate valid spawn locations
        valid_spawns = []
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                # Avoid rescue zone for spawning pits/obstacles
                if x < self.GRID_WIDTH - rescue_width:
                    valid_spawns.append((x, y))
        
        self.np_random.shuffle(valid_spawns)

        # Place Pits
        num_pits = min(len(valid_spawns), 5 + self.level * 2)
        for _ in range(num_pits):
            x, y = valid_spawns.pop()
            self.pits.append(pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

        # Place Oscillating Obstacles
        num_obstacles = min(len(valid_spawns), 2 + self.level)
        for _ in range(num_obstacles):
            x, y = valid_spawns.pop()
            axis = self.np_random.choice(['x', 'y'])
            move_range = self.np_random.integers(2, 5)
            if axis == 'x':
                min_coord, max_coord = max(0, x - move_range // 2), min(self.GRID_WIDTH - 1, x + move_range // 2)
            else: # axis == 'y'
                min_coord, max_coord = max(0, y - move_range // 2), min(self.GRID_HEIGHT - 1, y + move_range // 2)

            self.obstacles.append({
                'pos': [x, y],
                'axis': axis,
                'range': (min_coord, max_coord),
                'dir': self.np_random.choice([-1, 1]),
                'speed': 0.02 + self.level * 0.005,
                'progress': self.np_random.random()
            })

        # Place Robots
        occupied_coords = set((p.x // self.CELL_SIZE, p.y // self.CELL_SIZE) for p in self.pits)
        occupied_coords.update(set((o['pos'][0], o['pos'][1]) for o in self.obstacles))
        
        start_x_max = 3
        robot_spawns = []
        for x in range(start_x_max, -1, -1):
            for y in range(self.GRID_HEIGHT):
                if (x, y) not in occupied_coords:
                    robot_spawns.append((x, y))

        self.np_random.shuffle(robot_spawns)
        
        for i in range(3):
            if not robot_spawns: # Failsafe if no spawn points left
                pos = (i, self.GRID_HEIGHT // 2)
            else:
                pos = robot_spawns.pop(0)
            self.robots.append({'id': i, 'pos': list(pos), 'color': self.COLOR_ROBOTS[i], 'rescued': False})
            occupied_coords.add(pos)

    def _update_active_robots(self):
        self.active_robot_indices = [i for i, r in enumerate(self.robots) if not r['rescued']]
        if not self.active_robot_indices:
            self.active_robot_selector = 0
            return
        
        # Ensure selector is valid
        self.active_robot_selector = self.active_robot_selector % len(self.active_robot_indices)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        terminated = False

        self._update_obstacles()
        self._update_particles()

        if movement == 0:  # Switch robot
            if len(self.active_robot_indices) > 1:
                self.active_robot_selector = (self.active_robot_selector + 1) % len(self.active_robot_indices)
                self.selected_robot_idx = self.active_robot_indices[self.active_robot_selector]
                # # SFX: switch_robot.wav
                self._create_particles(self.robots[self.selected_robot_idx]['pos'], self.COLOR_SELECTION, 10, 2)
        else: # Move robot
            if not self.active_robot_indices:
                return self._get_observation(), 0, terminated, False, self._get_info()

            robot = self.robots[self.selected_robot_idx]
            if robot['rescued']:
                 return self._get_observation(), 0, terminated, False, self._get_info()

            current_pos = list(robot['pos'])
            target_pos = list(current_pos)
            
            if movement == 1: target_pos[1] -= 1  # Up
            elif movement == 2: target_pos[1] += 1 # Down
            elif movement == 3: target_pos[0] -= 1 # Left
            elif movement == 4: target_pos[0] += 1 # Right

            tx, ty = target_pos[0], target_pos[1]

            # --- Collision and Boundary Checks ---
            is_valid_move = True
            if not (0 <= tx < self.GRID_WIDTH and 0 <= ty < self.GRID_HEIGHT):
                is_valid_move = False # Wall collision

            if is_valid_move:
                # Check collision with other robots
                for i, r in enumerate(self.robots):
                    if i != self.selected_robot_idx and not r['rescued'] and r['pos'] == target_pos:
                        is_valid_move = False
                        break
                # Check collision with obstacles
                for o in self.obstacles:
                    if o['pos'] == target_pos:
                        is_valid_move = False
                        break
            
            if is_valid_move:
                # Check for pits (game over)
                for pit in self.pits:
                    if pit.collidepoint(tx * self.CELL_SIZE + self.CELL_SIZE/2, ty * self.CELL_SIZE + self.CELL_SIZE/2):
                        # # SFX: robot_fall.wav
                        self._create_particles(target_pos, robot['color'], 30, 4)
                        self.game_over = True
                        self.game_over_message = "A ROBOT FELL!"
                        reward -= 10
                        terminated = True
                        is_valid_move = False
                        break
            
            if is_valid_move:
                # Calculate distance-based reward
                dist_before = abs(current_pos[0] - self.rescue_zone_center[0]) + abs(current_pos[1] - self.rescue_zone_center[1])
                dist_after = abs(target_pos[0] - self.rescue_zone_center[0]) + abs(target_pos[1] - self.rescue_zone_center[1])
                reward += (dist_before - dist_after) * 0.1

                # Update position
                robot['pos'] = target_pos
                # # SFX: robot_move.wav
                
                # Check for rescue
                if self.rescue_zone.collidepoint(tx * self.CELL_SIZE + self.CELL_SIZE/2, ty * self.CELL_SIZE + self.CELL_SIZE/2):
                    robot['rescued'] = True
                    # # SFX: robot_rescued.wav
                    self._create_particles(target_pos, (0, 255, 0), 50, 5)
                    reward += 10
                    self._update_active_robots()
                    if not self.active_robot_indices: # All robots rescued
                        self.game_over = True
                        self.game_over_message = "ALL ROBOTS SAFE!"
                        reward += 100
                        terminated = True
                    else:
                        # Select next available robot
                        self.selected_robot_idx = self.active_robot_indices[self.active_robot_selector]

        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            self.game_over = True
            self.game_over_message = "TIME'S UP!"
            terminated = True
        
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_obstacles(self):
        for o in self.obstacles:
            o['progress'] += o['speed'] * o['dir']
            if not (0 <= o['progress'] <= 1):
                o['progress'] = np.clip(o['progress'], 0, 1)
                o['dir'] *= -1

            if o['axis'] == 'x':
                o['pos'][0] = o['range'][0] + int(round((o['range'][1] - o['range'][0]) * o['progress']))
            else:
                o['pos'][1] = o['range'][0] + int(round((o['range'][1] - o['range'][0]) * o['progress']))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _create_particles(self, grid_pos, color, count, max_speed):
        pixel_pos = [
            (grid_pos[0] + 0.5) * self.CELL_SIZE,
            (grid_pos[1] + 0.5) * self.CELL_SIZE
        ]
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * max_speed
            self.particles.append({
                'pos': list(pixel_pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(10, 20),
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw rescue zone
        s = pygame.Surface(self.rescue_zone.size, pygame.SRCALPHA)
        s.fill(self.COLOR_RESCUE_ZONE)
        self.screen.blit(s, self.rescue_zone.topleft)

        # Draw pits
        for pit in self.pits:
            pygame.draw.rect(self.screen, self.COLOR_PIT, pit)

        # Draw obstacles
        for o in self.obstacles:
            rect = pygame.Rect(o['pos'][0] * self.CELL_SIZE, o['pos'][1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect, border_radius=4)
        
        # Draw particles
        for p in self.particles:
            size = max(1, p['life'] / 5)
            pygame.draw.circle(self.screen, p['color'], p['pos'], size)
            
        # Draw robots
        for i, robot in enumerate(self.robots):
            if robot['rescued']:
                continue
            
            center_x = int((robot['pos'][0] + 0.5) * self.CELL_SIZE)
            center_y = int((robot['pos'][1] + 0.5) * self.CELL_SIZE)
            radius = int(self.CELL_SIZE * 0.35)

            # Selection highlight
            if i == self.selected_robot_idx and not self.game_over:
                pulse = abs(math.sin(self.steps * 0.3))
                sel_radius = radius + 3 + pulse * 3
                sel_color = self.COLOR_SELECTION
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, int(sel_radius), sel_color)

            # Robot body
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, robot['color'])
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, robot['color'])

    def _render_ui(self):
        # Score and rescued count
        rescued_count = sum(1 for r in self.robots if r['rescued'])
        rescued_text = self.font_ui.render(f"RESCUED: {rescued_count}/3", True, self.COLOR_TEXT)
        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        steps_text = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        
        self.screen.blit(rescued_text, (10, 10))
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        self.screen.blit(steps_text, (self.SCREEN_WIDTH // 2 - steps_text.get_width() // 2, 10))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_game_over.render(self.game_over_message, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "robots_rescued": sum(1 for r in self.robots if r['rescued']),
            "active_robots": len(self.active_robot_indices),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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
    env = GameEnv(level=1)
    env.reset()
    
    # --- Pygame Interactive Loop ---
    running = True
    terminated = False
    
    # Map pygame keys to actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    # Create a display for interactive playing
    pygame.display.set_caption("Robot Rescue")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    while running:
        action = np.array([0, 0, 0]) # Default action is no-op (switch robot)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    terminated = False
                    env.reset()
                
                if not terminated:
                    if event.key in key_to_action:
                        action[0] = key_to_action[event.key]
                    
                    # Step the environment with the determined action
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")

        # Update the display
        frame = env._get_observation()
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        display_screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit frame rate for interactive mode

    env.close()