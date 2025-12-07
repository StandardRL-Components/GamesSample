
# Generated: 2025-08-27T20:42:58.809080
# Source Brief: brief_02554.md
# Brief Index: 2554

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Use arrow keys to select a robot. Hold Space and press an arrow key to move it. Charge all robots to win!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A turn-based puzzle game where you guide stranded robots to their matching charging stations before running out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 16
    GRID_ROWS = 10
    TILE_SIZE = 40
    
    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (45, 50, 64)
    COLOR_WALL = (68, 76, 96)
    COLOR_TEXT = (220, 220, 220)
    COLOR_STATION_ICON = (255, 223, 0) # Gold
    
    ROBOT_COLORS = [
        (255, 87, 87),   # Red
        (87, 255, 150),  # Green
        (87, 150, 255),  # Blue
        (255, 255, 87),  # Yellow
        (255, 87, 255),  # Magenta
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = 0
        self.robots = []
        self.stations = []
        self.walls = []
        self.selected_robot_idx = 0
        self.particles = []
        self.rng = None
        
        # Initialize state
        self.reset()
        
        # Run self-check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = 50
        self.selected_robot_idx = 0
        self.particles = []
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        terminated = False

        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        # shift_held (action[2]) is ignored as per brief

        # --- Action Logic ---
        # 1. Selection Action (Arrow key without Space)
        if movement != 0 and not space_held:
            num_robots = len(self.robots)
            if movement in [1, 4]:  # Up or Right cycles forward
                self.selected_robot_idx = (self.selected_robot_idx + 1) % num_robots
            elif movement in [2, 3]:  # Down or Left cycles backward
                self.selected_robot_idx = (self.selected_robot_idx - 1 + num_robots) % num_robots
            # No reward or move cost for just selecting
            
        # 2. Movement Action (Arrow key WITH Space)
        elif movement != 0 and space_held:
            self.moves_left -= 1
            reward -= 0.1  # Cost for attempting a move

            robot = self.robots[self.selected_robot_idx]
            old_pos = robot['pos']
            
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(movement, (0, 0))
            new_pos = (old_pos[0] + dx, old_pos[1] + dy)

            if self._is_valid_move_target(new_pos):
                # Sound: robot_move.wav
                self._create_move_particles(old_pos, 0.5)
                robot['pos'] = new_pos

                was_on_station = robot['on_station']
                robot['on_station'] = False
                for station in self.stations:
                    if robot['pos'] == station['pos'] and robot['color'] == station['color']:
                        robot['on_station'] = True
                        if not was_on_station:
                            # Sound: charge_up.wav
                            reward += 1.0
                            self.score += 1
                            self._create_charge_particles(new_pos)
                        break
        
        # 3. No-op (all other cases)
        # No state change, no reward.
        
        terminated = self._check_termination()
        if terminated:
            if self._check_win_condition():
                # Sound: level_complete.wav
                reward += 10.0
                self.score += 10
            else: # Ran out of moves
                # Sound: game_over.wav
                reward -= 10.0
                self.score -= 10
            self.game_over = True
        
        self._update_particles()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_grid()
        self._draw_walls()
        self._draw_stations()
        self._draw_robots()
        self._draw_particles()
        self._draw_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "robots_charged": sum(1 for r in self.robots if r['on_station']),
        }

    def _generate_level(self):
        num_robots = self.rng.integers(3, 6)
        
        # Generate all possible grid positions
        all_positions = [(x, y) for x in range(self.GRID_COLS) for y in range(self.GRID_ROWS)]
        self.rng.shuffle(all_positions)
        
        # Place robots and stations
        self.robots = []
        self.stations = []
        robot_colors = list(self.ROBOT_COLORS)
        self.rng.shuffle(robot_colors)

        for i in range(num_robots):
            robot_pos = all_positions.pop()
            station_pos = all_positions.pop()
            color = robot_colors[i % len(robot_colors)]
            
            self.robots.append({'pos': robot_pos, 'color': color, 'on_station': False})
            self.stations.append({'pos': station_pos, 'color': color})
        
        # Place walls
        self.walls = []
        num_walls = self.rng.integers(10, 21)
        for _ in range(num_walls):
            if not all_positions: break
            self.walls.append(all_positions.pop())

    def _is_valid_move_target(self, pos):
        # Check grid boundaries
        if not (0 <= pos[0] < self.GRID_COLS and 0 <= pos[1] < self.GRID_ROWS):
            return False
        # Check wall collision
        if pos in self.walls:
            return False
        # Check other robot collision
        for robot in self.robots:
            if robot['pos'] == pos:
                return False
        return True

    def _check_termination(self):
        if self.moves_left <= 0:
            return True
        return self._check_win_condition()

    def _check_win_condition(self):
        return all(r['on_station'] for r in self.robots)

    def _grid_to_pixel(self, grid_pos):
        x = grid_pos[0] * self.TILE_SIZE + self.TILE_SIZE // 2
        y = grid_pos[1] * self.TILE_SIZE + self.TILE_SIZE // 2
        return int(x), int(y)

    def _draw_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _draw_walls(self):
        for wall_pos in self.walls:
            px, py = self._grid_to_pixel(wall_pos)
            rect = pygame.Rect(px - self.TILE_SIZE // 2, py - self.TILE_SIZE // 2, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)

    def _draw_stations(self):
        for station in self.stations:
            px, py = self._grid_to_pixel(station['pos'])
            color = station['color']
            
            # Draw base
            pygame.gfxdraw.box(self.screen, 
                               pygame.Rect(px - self.TILE_SIZE // 2, py - self.TILE_SIZE // 2, self.TILE_SIZE, self.TILE_SIZE),
                               (color[0]//4, color[1]//4, color[2]//4))

            # Draw lightning bolt icon
            bolt_points = [
                (px - 6, py + 10), (px + 6, py - 2), (px, py - 2),
                (px + 6, py - 10), (px - 6, py + 2), (px, py + 2)
            ]
            pygame.gfxdraw.aapolygon(self.screen, bolt_points, self.COLOR_STATION_ICON)
            pygame.gfxdraw.filled_polygon(self.screen, bolt_points, self.COLOR_STATION_ICON)

    def _draw_robots(self):
        for i, robot in enumerate(self.robots):
            px, py = self._grid_to_pixel(robot['pos'])
            radius = self.TILE_SIZE // 3
            
            # Selection highlight
            if i == self.selected_robot_idx and not self.game_over:
                pulse_radius = radius + 5 + 3 * math.sin(self.steps * 0.3)
                pygame.gfxdraw.aacircle(self.screen, px, py, int(pulse_radius), (255, 255, 255))

            # Robot body
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, robot['color'])
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, (0,0,0))
            
            # Robot "eye"
            eye_radius = radius // 3
            pygame.gfxdraw.filled_circle(self.screen, px, py, eye_radius, (255, 255, 255))
            pygame.gfxdraw.aacircle(self.screen, px, py, eye_radius, (0,0,0))

            if robot['on_station']:
                # Charged indicator (glowing outline)
                pygame.gfxdraw.aacircle(self.screen, px, py, radius + 2, self.COLOR_STATION_ICON)

    def _draw_ui(self):
        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 15, 10))

        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            status_text_str = "VICTORY!" if self._check_win_condition() else "OUT OF MOVES"
            status_text = self.font_large.render(status_text_str, True, self.COLOR_STATION_ICON)
            text_rect = status_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(status_text, text_rect)

    # --- Particle System ---
    def _create_move_particles(self, grid_pos, speed_mult=1.0):
        px, py = self._grid_to_pixel(grid_pos)
        for _ in range(10):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 2 * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.rng.integers(15, 30)
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': life, 'color': self.COLOR_GRID})

    def _create_charge_particles(self, grid_pos):
        px, py = self._grid_to_pixel(grid_pos)
        robot = next(r for r in self.robots if r['pos'] == grid_pos)
        for _ in range(30):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 3
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.rng.integers(20, 40)
            color = self.rng.choice([self.COLOR_STATION_ICON, robot['color'], (255,255,255)])
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': life, 'color': color})

    def _update_particles(self):
        if not self.particles: return
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _draw_particles(self):
        for p in self.particles:
            size = max(0, p['life'] // 5)
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), size)

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0]  # Default no-op action
        
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        
        # Selection is handled by tapping an arrow key
        # Movement is handled by holding space and pressing an arrow key
        movement_key_pressed = False
        if keys[pygame.K_UP]:
            action[0] = 1
            movement_key_pressed = True
        elif keys[pygame.K_DOWN]:
            action[0] = 2
            movement_key_pressed = True
        elif keys[pygame.K_LEFT]:
            action[0] = 3
            movement_key_pressed = True
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            movement_key_pressed = True

        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        # Event handling
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                # Selection happens on key down if space is not held
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT] and not keys[pygame.K_SPACE]:
                    action_taken = True
                # Movement happens on key down if space is held
                elif event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT] and keys[pygame.K_SPACE]:
                    action_taken = True
                elif event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
                    action_taken = False # Don't step on reset
                
        if action_taken:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
            if terminated:
                print("Game Over! Press 'R' to restart.")

        # --- Rendering ---
        # The environment already renders to a surface, so we just blit it
        rendered_frame = np.transpose(obs, (1, 0, 2))
        pygame.surfarray.blit_array(screen, rendered_frame)
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate

    pygame.quit()