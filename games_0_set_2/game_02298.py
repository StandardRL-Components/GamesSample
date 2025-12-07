
# Generated: 2025-08-27T19:57:32.702044
# Source Brief: brief_02298.md
# Brief Index: 2298

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a Sokoban-style puzzle game with a time limit.

    The player must push all crates onto their designated target locations before
    the timer runs out. The game features smooth, interpolated graphics for a
    polished arcade feel.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move. Push all the crates onto the red targets before time runs out."
    )

    game_description = (
        "A timed puzzle game where you must push crates onto their designated targets. Plan your moves carefully to solve the level before the clock hits zero."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 20
    GRID_HEIGHT = 12
    CELL_SIZE = 32  # Keep it clean for pixel art feel
    
    GAME_AREA_WIDTH = GRID_WIDTH * CELL_SIZE
    GAME_AREA_HEIGHT = GRID_HEIGHT * CELL_SIZE
    
    # Center the game area
    GRID_OFFSET_X = (SCREEN_WIDTH - GAME_AREA_WIDTH) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GAME_AREA_HEIGHT) // 2

    FPS = 60
    MAX_TIME = 60  # seconds
    MAX_STEPS = MAX_TIME * FPS

    # Visuals
    ANIM_SPEED = 0.2  # Speed of visual interpolation (lower is faster)
    
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (35, 40, 60)
    COLOR_WALL = (80, 90, 110)
    COLOR_PLAYER = (50, 200, 255)
    COLOR_PLAYER_GLOW = (50, 200, 255, 50)
    COLOR_CRATE = (160, 110, 80)
    COLOR_TARGET = (255, 50, 100)
    COLOR_CRATE_ON_TARGET = (110, 200, 90)
    COLOR_TARGET_FILLED = (110, 200, 90)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TIMER_WARN = (255, 200, 0)
    COLOR_TIMER_CRIT = (255, 50, 50)
    
    PARTICLE_COLORS = [(255, 200, 80), (255, 150, 50), (255, 255, 150)]

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
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        # Game state variables are initialized in reset()
        self.level_layout = []
        self.player_pos = (0, 0)
        self.crate_positions = []
        self.target_positions = []
        
        # Visual state for smooth animation
        self.player_visual_pos = [0.0, 0.0]
        self.crates_visual_pos = []
        self.is_animating = False

        self.time_remaining = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.particles = []

        self.reset()
        
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._setup_level()

        self.steps = 0
        self.score = 0
        self.time_remaining = self.MAX_TIME
        self.game_over = False
        self.win_message = ""
        self.is_animating = False
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.01  # Small penalty for each step to encourage speed

        crates_on_target_before = self._count_crates_on_targets()

        if not self.is_animating and movement != 0:
            self._handle_movement(movement)

        self._update_animations()

        crates_on_target_after = self._count_crates_on_targets()
        
        # Reward for moving a crate onto a target
        if crates_on_target_after > crates_on_target_before:
            # sfx: crate_on_target_sound.play()
            reward += 1.0 * (crates_on_target_after - crates_on_target_before)
            
            # Find which crate just landed and create particles
            for i, pos in enumerate(self.crate_positions):
                if pos in self.target_positions and self.crates_visual_pos[i] != self._grid_to_pixel(pos):
                     self._spawn_particles(self._grid_to_pixel(pos))

        # Reward for moving a crate off a target (discourage mistakes)
        if crates_on_target_after < crates_on_target_before:
            reward -= 1.0 * (crates_on_target_before - crates_on_target_after)

        self.time_remaining -= 1 / self.FPS
        self.steps += 1
        self.score = crates_on_target_after
        
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.score == len(self.target_positions):
                reward += 100
                self.win_message = "LEVEL COMPLETE!"
            else:
                reward -= 100
                self.win_message = "TIME'S UP!"

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _setup_level(self):
        """Initializes the game state for a new level."""
        level_map = [
            "####################",
            "#                  #",
            "# P       C        #",
            "# # # # # # #      #",
            "#         # C    T #",
            "# C T     # # # ## #",
            "#         #   C  T #",
            "# ###########      #",
            "#   T              #",
            "#                  #",
            "#                  #",
            "####################",
        ]
        
        self.level_layout = [[(1 if char == '#' else 0) for char in row] for row in level_map]
        
        self.crate_positions = []
        self.target_positions = []

        for y, row in enumerate(level_map):
            for x, char in enumerate(row):
                if char == 'P':
                    self.player_pos = (x, y)
                elif char == 'C':
                    self.crate_positions.append((x, y))
                elif char == 'T':
                    self.target_positions.append((x, y))
        
        # Initialize visual positions to match logical positions
        self.player_visual_pos = list(self._grid_to_pixel(self.player_pos))
        self.crates_visual_pos = [list(self._grid_to_pixel(pos)) for pos in self.crate_positions]

    def _handle_movement(self, movement):
        """Processes player movement and crate pushing logic."""
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right

        if dx == 0 and dy == 0:
            return

        next_player_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)

        # Check for wall collision
        if self.level_layout[next_player_pos[1]][next_player_pos[0]] == 1:
            # sfx: bump_wall_sound.play()
            return

        # Check for crate collision
        if next_player_pos in self.crate_positions:
            crate_index = self.crate_positions.index(next_player_pos)
            next_crate_pos = (next_player_pos[0] + dx, next_player_pos[1] + dy)

            # Check if crate can be pushed (not into a wall or another crate)
            if self.level_layout[next_crate_pos[1]][next_crate_pos[0]] == 1 or next_crate_pos in self.crate_positions:
                # sfx: bump_crate_sound.play()
                return
            
            # Push crate
            self.crate_positions[crate_index] = next_crate_pos
            # sfx: push_crate_sound.play()
        
        # Move player
        self.player_pos = next_player_pos
        self.is_animating = True

    def _update_animations(self):
        """Interpolates visual positions towards logical positions."""
        is_still_animating = False
        
        # Animate player
        target_px = self._grid_to_pixel(self.player_pos)
        current_px = self.player_visual_pos
        dist_sq = (target_px[0] - current_px[0])**2 + (target_px[1] - current_px[1])**2
        
        if dist_sq > 1:
            current_px[0] += (target_px[0] - current_px[0]) * self.ANIM_SPEED
            current_px[1] += (target_px[1] - current_px[1]) * self.ANIM_SPEED
            is_still_animating = True
        else:
            self.player_visual_pos = list(target_px)

        # Animate crates
        for i, pos in enumerate(self.crate_positions):
            target_px = self._grid_to_pixel(pos)
            current_px = self.crates_visual_pos[i]
            dist_sq = (target_px[0] - current_px[0])**2 + (target_px[1] - current_px[1])**2

            if dist_sq > 1:
                current_px[0] += (target_px[0] - current_px[0]) * self.ANIM_SPEED
                current_px[1] += (target_px[1] - current_px[1]) * self.ANIM_SPEED
                is_still_animating = True
            else:
                self.crates_visual_pos[i] = list(target_px)
        
        self.is_animating = is_still_animating
        
        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        """Checks for win/loss conditions."""
        all_crates_on_targets = self._count_crates_on_targets() == len(self.target_positions)
        time_up = self.time_remaining <= 0
        return all_crates_on_targets or time_up or self.steps >= self.MAX_STEPS

    def _calculate_reward(self):
        # This logic is now integrated directly into step() for clarity
        pass

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
            "time_remaining": self.time_remaining,
            "crates_on_target": self._count_crates_on_targets(),
            "total_crates": len(self.crate_positions)
        }

    def _grid_to_pixel(self, grid_pos):
        """Converts grid coordinates to pixel coordinates."""
        x = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE
        y = self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE
        return (x, y)

    def _render_game(self):
        """Renders all game world elements."""
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            px = self.GRID_OFFSET_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_OFFSET_Y), (px, self.GRID_OFFSET_Y + self.GAME_AREA_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.GRID_OFFSET_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, py), (self.GRID_OFFSET_X + self.GAME_AREA_WIDTH, py))

        # Draw walls
        for y, row in enumerate(self.level_layout):
            for x, cell in enumerate(row):
                if cell == 1:
                    px, py = self._grid_to_pixel((x, y))
                    rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
        
        # Draw targets
        for pos in self.target_positions:
            px, py = self._grid_to_pixel(pos)
            center = (px + self.CELL_SIZE // 2, py + self.CELL_SIZE // 2)
            color = self.COLOR_TARGET_FILLED if pos in self.crate_positions else self.COLOR_TARGET
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], self.CELL_SIZE // 4, color)
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], self.CELL_SIZE // 4, color)

        # Draw crates
        for i, pos in enumerate(self.crate_positions):
            px, py = self.crates_visual_pos[i]
            rect = pygame.Rect(px + 4, py + 4, self.CELL_SIZE - 8, self.CELL_SIZE - 8)
            color = self.COLOR_CRATE_ON_TARGET if pos in self.target_positions else self.COLOR_CRATE
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            pygame.draw.rect(self.screen, tuple(c*0.8 for c in color), rect, width=2, border_radius=4)
        
        # Draw player
        px, py = self.player_visual_pos
        center = (int(px + self.CELL_SIZE / 2), int(py + self.CELL_SIZE / 2))
        radius = self.CELL_SIZE // 2 - 4
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius + 3, self.COLOR_PLAYER_GLOW)
        # Player body
        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, self.COLOR_PLAYER)
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = p['color'] + (alpha,)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                pygame.draw.circle(self.screen, color, (int(p['pos'][0]), int(p['pos'][1])), size)

    def _render_ui(self):
        """Renders UI elements like score and timer."""
        # Render score / crates on target
        score_text = f"COMPLETED: {self.score}/{len(self.target_positions)}"
        score_surf = self.font_medium.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 10))

        # Render timer
        time_left = max(0, self.time_remaining)
        timer_text = f"{int(time_left // 60):02}:{int(time_left % 60):02}"
        timer_color = self.COLOR_TEXT
        if time_left < 10: timer_color = self.COLOR_TIMER_CRIT
        elif time_left < 20: timer_color = self.COLOR_TIMER_WARN
        
        timer_surf = self.font_large.render(timer_text, True, timer_color)
        timer_rect = timer_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 5))
        self.screen.blit(timer_surf, timer_rect)

        # Render game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            win_surf = self.font_large.render(self.win_message, True, self.COLOR_TEXT)
            win_rect = win_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(win_surf, win_rect)

    def _count_crates_on_targets(self):
        """Counts how many crates are currently on target locations."""
        return sum(1 for pos in self.crate_positions if pos in self.target_positions)

    def _spawn_particles(self, pos):
        """Creates a burst of particles at a given pixel position."""
        center_pos = (pos[0] + self.CELL_SIZE // 2, pos[1] + self.CELL_SIZE // 2)
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            life = random.randint(15, 30)
            self.particles.append({
                'pos': list(center_pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': random.choice(self.PARTICLE_COLORS),
                'size': random.randint(2, 5)
            })

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    terminated = False
    
    # Use a dummy window for rendering if this script is run directly
    pygame.display.set_caption("Sokoban Timer")
    real_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    while running:
        action = [0, 0, 0] # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        if keys[pygame.K_r]: # Reset on 'r' key
            obs, info = env.reset()
            terminated = False
            continue
            
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Blit the environment's rendered surface to the real screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        real_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(env.FPS)
        
    env.close()