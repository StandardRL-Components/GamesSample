
# Generated: 2025-08-28T05:57:36.818858
# Source Brief: brief_05736.md
# Brief Index: 5736

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move selected robot. Spacebar to cycle selection. Each action costs 1 move."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide 8 stranded robots to their charging stations. Plan your moves carefully, as you have a limited number."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.GRID_SIZE = 10
        self.NUM_ROBOTS = 8
        self.NUM_OBSTACLES = 2
        self.MAX_MOVES = 100
        self.MAX_STEPS = 1000
        
        self.WIDTH, self.HEIGHT = 640, 400
        
        # Grid rendering properties
        self.TILE_SIZE = 40
        self.GRID_WIDTH = self.GRID_SIZE * self.TILE_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.TILE_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2
        
        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_STATION = (255, 220, 0)
        self.COLOR_OBSTACLE = (100, 110, 120)
        self.COLOR_TEXT = (230, 240, 250)
        self.ROBOT_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
            (80, 255, 255),  # Cyan
            (255, 80, 255),  # Magenta
            (255, 160, 80),  # Orange
            (160, 80, 255),  # Purple
            (255, 105, 180)  # Pink
        ]

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Initialize state variables
        self.robot_pos = []
        self.station_pos = []
        self.obstacle_pos = []
        self.robots_charged = []
        self.selected_robot_idx = 0
        self.previous_space_held = False
        self.moves_remaining = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = self.MAX_MOVES
        self.selected_robot_idx = 0
        self.previous_space_held = False
        
        self._generate_level()
        
        self.robots_charged = [False] * self.NUM_ROBOTS
        self._update_charge_status()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        # Generate a valid, non-overlapping level layout
        all_coords = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
        self.np_random.shuffle(all_coords)

        # Place stations
        self.station_pos = [list(all_coords.pop()) for _ in range(self.NUM_ROBOTS)]

        # Place robots
        self.robot_pos = [list(all_coords.pop()) for _ in range(self.NUM_ROBOTS)]
            
        # Place obstacles
        self.obstacle_pos = []
        # Filter remaining coordinates to ensure obstacles are not adjacent to start/end points
        valid_obstacle_coords = [
            p for p in all_coords
            if min(self._dist(p, sp) for sp in self.station_pos) > 1 and \
               min(self._dist(p, rp) for rp in self.robot_pos) > 1
        ]
        self.np_random.shuffle(valid_obstacle_coords)
        
        for _ in range(self.NUM_OBSTACLES):
            if not valid_obstacle_coords: break
            self.obstacle_pos.append(list(valid_obstacle_coords.pop()))

    def _dist(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # Every step costs one move
        self.moves_remaining -= 1

        movement = action[0]
        space_held = action[1] == 1
        
        # 1. Handle selection change (on key press)
        if space_held and not self.previous_space_held:
            self.selected_robot_idx = (self.selected_robot_idx + 1) % self.NUM_ROBOTS
            reward -= 0.2  # Penalty for a non-movement action
        
        # 2. Handle movement (if not changing selection this step)
        elif movement != 0:
            robot_idx = self.selected_robot_idx
            
            # Can't move a robot that is already charged
            if self.robots_charged[robot_idx]:
                reward -= 0.2
            else:
                current_pos = self.robot_pos[robot_idx]
                
                # Calculate distance to nearest available station before move
                old_dist = self._get_dist_to_nearest_station(robot_idx)
                
                dx, dy = 0, 0
                if movement == 1: dy = -1  # Up
                elif movement == 2: dy = 1   # Down
                elif movement == 3: dx = -1  # Left
                elif movement == 4: dx = 1   # Right
                
                new_pos = [current_pos[0] + dx, current_pos[1] + dy]

                # Check for validity of the new position
                if self._is_valid_pos(new_pos, robot_idx):
                    # sound: robot_move_click
                    self.robot_pos[robot_idx] = new_pos
                    new_dist = self._get_dist_to_nearest_station(robot_idx)
                    
                    if new_dist < old_dist:
                        reward += 1.0  # Moved closer
                    else:
                        reward -= 0.2 # Moved away or parallel
                else:
                    # sound: bump_sfx
                    reward -= 0.2 # Penalty for invalid move
        else:
            # No movement and no selection change is a "wait" action
            reward -= 0.2

        self.previous_space_held = space_held
        
        # 3. Update game state and check for terminal conditions
        newly_charged_count = self._update_charge_status()
        if newly_charged_count > 0:
            # sound: bleep_charge_up!
            reward += newly_charged_count * 10
            self.score += newly_charged_count * 10
        
        terminated = False
        if all(self.robots_charged):
            # sound: victory_fanfare!
            reward += 100
            self.score += 100
            terminated = True
            self.game_over = True
        elif self.moves_remaining <= 0:
            # sound: failure_buzzer!
            reward -= 100
            self.score -= 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_dist_to_nearest_station(self, robot_idx):
        robot_pos = self.robot_pos[robot_idx]
        other_robot_positions = {tuple(pos) for i, pos in enumerate(self.robot_pos) if i != robot_idx}
        
        available_stations = [
            s_pos for s_pos in self.station_pos if tuple(s_pos) not in other_robot_positions
        ]
        
        if not available_stations:
            return float('inf')
        
        return min(self._dist(robot_pos, s_pos) for s_pos in available_stations)

    def _is_valid_pos(self, pos, moving_robot_idx):
        # Check grid bounds
        if not (0 <= pos[0] < self.GRID_SIZE and 0 <= pos[1] < self.GRID_SIZE):
            return False
        
        # Check obstacle collision
        if pos in self.obstacle_pos:
            return False
            
        # Check other robot collision
        for i, other_pos in enumerate(self.robot_pos):
            if i != moving_robot_idx and pos == other_pos:
                return False
        
        return True

    def _update_charge_status(self):
        newly_charged_count = 0
        station_coords_set = {tuple(pos) for pos in self.station_pos}
        for i in range(self.NUM_ROBOTS):
            is_charged = tuple(self.robot_pos[i]) in station_coords_set
            if is_charged and not self.robots_charged[i]:
                newly_charged_count += 1
            self.robots_charged[i] = is_charged
        return newly_charged_count

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _grid_to_pixels(self, pos):
        x = self.GRID_OFFSET_X + pos[0] * self.TILE_SIZE + self.TILE_SIZE // 2
        y = self.GRID_OFFSET_Y + pos[1] * self.TILE_SIZE + self.TILE_SIZE // 2
        return int(x), int(y)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_v = (self.GRID_OFFSET_X + i * self.TILE_SIZE, self.GRID_OFFSET_Y)
            end_v = (self.GRID_OFFSET_X + i * self.TILE_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_v, end_v, 1)
            # Horizontal
            start_h = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.TILE_SIZE)
            end_h = (self.GRID_OFFSET_X + self.GRID_WIDTH, self.GRID_OFFSET_Y + i * self.TILE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_h, end_h, 1)

        # Draw stations
        for pos in self.station_pos:
            px, py = self._grid_to_pixels(pos)
            size = self.TILE_SIZE // 4
            pygame.draw.rect(self.screen, self.COLOR_STATION, (px - size, py - size, size * 2, size * 2), border_radius=3)
            
        # Draw obstacles
        for pos in self.obstacle_pos:
            px, py = self._grid_to_pixels(pos)
            size = self.TILE_SIZE // 2 - 2
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, (px - size, py - size, size * 2, size * 2))

        # Draw robots
        for i, pos in enumerate(self.robot_pos):
            px, py = self._grid_to_pixels(pos)
            color = self.ROBOT_COLORS[i]
            
            # Selection glow
            if i == self.selected_robot_idx and not self.game_over:
                glow_alpha = 100 + 50 * math.sin(self.steps * 0.2)
                glow_color = (*color, glow_alpha)
                glow_radius = int(self.TILE_SIZE // 2 * 0.9)
                temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, glow_radius, glow_radius, glow_radius, glow_color)
                pygame.gfxdraw.aacircle(temp_surf, glow_radius, glow_radius, glow_radius, glow_color)
                self.screen.blit(temp_surf, (px - glow_radius, py - glow_radius))

            # Main robot circle
            radius = int(self.TILE_SIZE / 3.5)
            # Charged pulse effect
            if self.robots_charged[i]:
                pulse = 2 * math.sin(self.steps * 0.3)
                radius = int(radius + pulse)
            
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, color)
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, color)

    def _render_ui(self):
        # Moves remaining
        moves_text = self.font_main.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 10))

        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Charged count
        charged_count = sum(self.robots_charged)
        charged_text = self.font_main.render(f"Charged: {charged_count} / {self.NUM_ROBOTS}", True, self.COLOR_TEXT)
        self.screen.blit(charged_text, ((self.WIDTH - charged_text.get_width()) // 2, self.HEIGHT - 30))
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if all(self.robots_charged):
                msg = "ALL ROBOTS CHARGED!"
                color = self.COLOR_STATION
            else:
                msg = "OUT OF MOVES"
                color = self.ROBOT_COLORS[0]
                
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "robots_charged": sum(self.robots_charged),
        }
    
    def close(self):
        pygame.font.quit()
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

# Example of how to run the environment for human play
if __name__ == '__main__':
    import sys
    import os
    
    # To see the game window, ensure SDL_VIDEODRIVER is not "dummy"
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    pygame.display.init()
    pygame.display.set_caption("Robot Charge Puzzle")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    running = True
    while running:
        movement = 0
        space_press = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                if event.key == pygame.K_q:
                    running = False
                
                # Handle single key presses for turn-based action
                if not terminated:
                    if event.key == pygame.K_UP: movement = 1
                    elif event.key == pygame.K_DOWN: movement = 2
                    elif event.key == pygame.K_LEFT: movement = 3
                    elif event.key == pygame.K_RIGHT: movement = 4
                    elif event.key == pygame.K_SPACE: space_press = True
                    
                    # An action is taken on any valid key press
                    action_taken = movement != 0 or space_press
                    if action_taken:
                        action = [movement, 1 if space_press else 0, 0]
                        obs, reward, terminated, truncated, info = env.step(action)

        # Draw the latest observation to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(60)

    env.close()
    sys.exit()