
# Generated: 2025-08-27T19:09:17.837785
# Source Brief: brief_02070.md
# Brief Index: 2070

        
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
        "Controls: Use arrow keys to move the crystal. Avoid the red lasers."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a crystal through laser-filled caverns to reach the green exit. You have a limited number of shields and steps."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 10, 40)
    COLOR_WALL = (50, 40, 80)
    COLOR_WALL_TOP = (80, 70, 110)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 100, 200)
    COLOR_EXIT = (0, 255, 100)
    COLOR_EXIT_GLOW = (0, 200, 80)
    COLOR_LASER = (255, 0, 50)
    COLOR_LASER_GLOW = (200, 0, 40)
    COLOR_PARTICLE = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (20, 20, 30)
    
    # Screen and Grid
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 25
    GRID_HEIGHT = 16
    TILE_WIDTH = 32
    TILE_HEIGHT = 16
    ISO_ORIGIN_X = SCREEN_WIDTH // 2
    ISO_ORIGIN_Y = 60
    
    # Game parameters
    MAX_STEPS = 1000
    INITIAL_HITS = 3
    BASE_LASER_SPEED = 0.05 # rad/step
    LEVEL_SPEED_INCREASE = 0.02 # rad/step
    
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
        
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_level = pygame.font.SysFont("Consolas", 24, bold=True)
        
        self.level = 1
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if options and "level" in options:
            self.level = options["level"]

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.hits_remaining = self.INITIAL_HITS
        
        self._setup_level()
        
        return self._get_observation(), self._get_info()
    
    def _setup_level(self):
        """Initializes the game state for the current level."""
        self.player_pos = np.array([1.0, 1.0])
        self.exit_pos = np.array([self.GRID_WIDTH - 2, self.GRID_HEIGHT - 2], dtype=float)
        
        # Generate walls
        self.walls = set()
        for x in range(self.GRID_WIDTH):
            self.walls.add((x, 0))
            self.walls.add((x, self.GRID_HEIGHT - 1))
        for y in range(self.GRID_HEIGHT):
            self.walls.add((0, y))
            self.walls.add((self.GRID_WIDTH - 1, y))

        # Add some internal obstacles
        for i in range(3 + self.level):
             # Ensure obstacles don't block start/end
            ox = self.np_random.integers(3, self.GRID_WIDTH - 3)
            oy = self.np_random.integers(3, self.GRID_HEIGHT - 3)
            if math.hypot(ox - self.player_pos[0], oy - self.player_pos[1]) > 3 and \
               math.hypot(ox - self.exit_pos[0], oy - self.exit_pos[1]) > 3:
                self.walls.add((ox, oy))

        # Generate lasers
        self.lasers = []
        num_lasers = min(5, self.level) # Cap lasers at 5
        for _ in range(num_lasers):
            while True:
                pivot = (
                    self.np_random.integers(1, self.GRID_WIDTH - 1),
                    self.np_random.integers(1, self.GRID_HEIGHT - 1)
                )
                if pivot not in self.walls:
                    break
            
            self.lasers.append({
                "pivot": np.array(pivot, dtype=float),
                "angle": self.np_random.uniform(0, 2 * math.pi),
                "speed": self.BASE_LASER_SPEED + (self.level - 1) * self.LEVEL_SPEED_INCREASE,
                "length": self.np_random.integers(4, 8)
            })
            
        self.particles = []

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.01 # Small penalty for taking a step
        
        # --- 1. Player Movement ---
        target_pos = self.player_pos.copy()
        if movement == 1: target_pos[1] -= 1 # Up
        elif movement == 2: target_pos[1] += 1 # Down
        elif movement == 3: target_pos[0] -= 1 # Left
        elif movement == 4: target_pos[0] += 1 # Right
        
        if tuple(target_pos) not in self.walls:
            self.player_pos = target_pos

        # --- 2. Update Game State ---
        self.steps += 1
        for laser in self.lasers:
            laser["angle"] = (laser["angle"] + laser["speed"]) % (2 * math.pi)

        # --- 3. Collision Detection ---
        player_hit = False
        player_center_iso = self._iso_transform(self.player_pos[0], self.player_pos[1])
        
        for laser in self.lasers:
            p1 = self._iso_transform(laser["pivot"][0], laser["pivot"][1])
            p2_grid = laser["pivot"] + laser["length"] * np.array([math.cos(laser["angle"]), math.sin(laser["angle"])])
            p2 = self._iso_transform(p2_grid[0], p2_grid[1])
            
            if self._line_point_collision(p1, p2, player_center_iso, self.TILE_HEIGHT / 2):
                player_hit = True
                break
        
        if player_hit:
            self.hits_remaining -= 1
            reward -= 10
            # sfx: player_hit_sound
            self._spawn_particles(player_center_iso, 20)

        # --- 4. Termination Check ---
        terminated = False
        if tuple(self.player_pos) == tuple(self.exit_pos):
            reward += 100
            if self.steps < 300: # Fast completion bonus
                reward += 20
            self.score += int(reward)
            self.level += 1
            terminated = True
            self.game_over = True
            # sfx: level_complete_sound
        elif self.hits_remaining <= 0:
            reward -= 50 # Extra penalty for dying
            terminated = True
            self.game_over = True
            # sfx: game_over_sound
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

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
            "level": self.level,
            "hits_remaining": self.hits_remaining
        }

    # --- Rendering Methods ---
    def _iso_transform(self, grid_x, grid_y):
        screen_x = self.ISO_ORIGIN_X + (grid_x - grid_y) * (self.TILE_WIDTH / 2)
        screen_y = self.ISO_ORIGIN_Y + (grid_x + grid_y) * (self.TILE_HEIGHT / 2)
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, pos, color, top_color, height=1):
        x, y = pos
        sx, sy = self._iso_transform(x, y)
        
        hw = self.TILE_WIDTH / 2
        hh = self.TILE_HEIGHT / 2
        
        points_top = [
            (sx, sy - height * hh),
            (sx + hw, sy - height * hh + hh),
            (sx, sy - height * hh + hh * 2),
            (sx - hw, sy - height * hh + hh),
        ]
        
        points_left = [
            (sx - hw, sy + hh), (sx - hw, sy - height * hh + hh),
            (sx, sy - height * hh + hh * 2), (sx, sy + hh * 2)
        ]

        points_right = [
            (sx + hw, sy + hh), (sx + hw, sy - height * hh + hh),
            (sx, sy - height * hh + hh * 2), (sx, sy + hh * 2)
        ]

        pygame.gfxdraw.filled_polygon(self.screen, points_right, [c * 0.7 for c in color])
        pygame.gfxdraw.filled_polygon(self.screen, points_left, [c * 0.5 for c in color])
        pygame.gfxdraw.filled_polygon(self.screen, points_top, top_color)
        pygame.gfxdraw.aapolygon(self.screen, points_top, top_color)

    def _render_game(self):
        # Sort elements by grid y for correct isometric drawing order
        render_queue = []
        for x, y in self.walls:
            render_queue.append(('wall', (x, y)))
        
        render_queue.append(('exit', self.exit_pos))
        render_queue.append(('player', self.player_pos))
        
        render_queue.sort(key=lambda item: item[1][0] + item[1][1])
        
        # Render sorted elements
        for type, pos in render_queue:
            if type == 'wall':
                self._draw_iso_cube(pos, self.COLOR_WALL, self.COLOR_WALL_TOP)
            elif type == 'exit':
                self._draw_iso_cube(pos, self.COLOR_EXIT_GLOW, self.COLOR_EXIT, height=0.1)
            elif type == 'player':
                self._render_player()

        self._render_lasers()
        self._update_and_render_particles()

    def _render_player(self):
        sx, sy = self._iso_transform(self.player_pos[0], self.player_pos[1])
        hw = self.TILE_WIDTH / 2
        hh = self.TILE_HEIGHT / 2
        
        points = [
            (sx, sy), (sx + hw, sy + hh), (sx, sy + hh * 2), (sx - hw, sy + hh)
        ]
        
        # Glow effect
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER_GLOW)
        
        # Main crystal
        points_inner = [
            (sx, sy + hh * 0.2), (sx + hw * 0.8, sy + hh), (sx, sy + hh * 1.8), (sx - hw * 0.8, sy + hh)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points_inner, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, points_inner, self.COLOR_PLAYER)

    def _render_lasers(self):
        for laser in self.lasers:
            p1 = self._iso_transform(laser["pivot"][0], laser["pivot"][1])
            p2_grid = laser["pivot"] + laser["length"] * np.array([math.cos(laser["angle"]), math.sin(laser["angle"])])
            p2 = self._iso_transform(p2_grid[0], p2_grid[1])
            
            # Draw glowing line
            pygame.draw.aaline(self.screen, self.COLOR_LASER_GLOW, p1, p2, 3)
            pygame.draw.aaline(self.screen, self.COLOR_LASER, p1, p2, 1)
            
            # Draw pivot
            pygame.gfxdraw.filled_circle(self.screen, p1[0], p1[1], 4, self.COLOR_LASER)
            pygame.gfxdraw.aacircle(self.screen, p1[0], p1[1], 4, self.COLOR_LASER)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            main = font.render(text, True, color)
            self.screen.blit(shadow, (pos[0] + 1, pos[1] + 1))
            self.screen.blit(main, pos)

        # Hits remaining
        hits_text = f"SHIELDS: {'● ' * self.hits_remaining}{'○ ' * (self.INITIAL_HITS - self.hits_remaining)}"
        draw_text(hits_text, self.font_ui, self.COLOR_TEXT, (self.SCREEN_WIDTH - 200, 10))

        # Steps remaining
        steps_text = f"STEPS: {self.steps}/{self.MAX_STEPS}"
        draw_text(steps_text, self.font_ui, self.COLOR_TEXT, (10, 10))
        
        # Level
        level_text = f"CAVERN {self.level}"
        text_surf = self.font_level.render(level_text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 25))
        shadow_surf = self.font_level.render(level_text, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)
        
        # Score
        score_text = f"SCORE: {int(self.score)}"
        draw_text(score_text, self.font_ui, self.COLOR_TEXT, (self.SCREEN_WIDTH // 2 - 60, 10))


    # --- Particle System ---
    def _spawn_particles(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(10, 20),
                'size': self.np_random.integers(2, 5)
            })

    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p['life'] / 20))))
                color = (*self.COLOR_PARTICLE, alpha)
                pygame.draw.circle(self.screen, color, p['pos'], p['size'])

    # --- Collision Logic ---
    def _line_point_collision(self, p1, p2, point, radius):
        p1 = np.array(p1)
        p2 = np.array(p2)
        point = np.array(point)
        
        d_p2_p1 = np.linalg.norm(p2 - p1)
        if d_p2_p1 == 0: return False

        t = np.dot(point - p1, p2 - p1) / (d_p2_p1**2)
        t = np.clip(t, 0, 1)
        
        closest_point = p1 + t * (p2 - p1)
        distance = np.linalg.norm(point - closest_point)
        
        return distance < radius

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

# Example of how to run the environment
if __name__ == '__main__':
    # Set dummy video driver for headless execution
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv()
    obs, info = env.reset()
    print("Initial state:", info)
    
    # Test a few random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Terminated={terminated}, Info={info}")
        if terminated:
            print("Episode finished.")
            obs, info = env.reset()
            print("Environment reset.")
    
    # Example of running with a display
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Crystal Caverns")
        clock = pygame.time.Clock()
        
        running = True
        while running:
            movement = 0 # no-op
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            action = [movement, 0, 0] # Space/Shift not used
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Draw the observation to the display screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated:
                print(f"Game Over! Final Score: {info['score']}. Resetting...")
                pygame.time.wait(2000)
                obs, info = env.reset()

            clock.tick(15) # Control human play speed

        pygame.quit()
    except pygame.error as e:
        print("\nPygame display could not be initialized. Human play is disabled.")
        print(f"Error: {e}")