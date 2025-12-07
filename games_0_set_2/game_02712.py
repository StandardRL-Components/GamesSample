import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Hold SHIFT and use arrow keys to move the cursor. "
        "Release SHIFT and use arrow keys to swap the selected crystal with an adjacent one."
    )

    game_description = (
        "An isometric puzzle game. Swap adjacent crystals to create matches of three or more. "
        "Plan your moves to create chain reactions and clear 20 crystals before you run out of moves."
    )

    auto_advance = True

    # --- Constants ---
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    NUM_COLORS = 3
    CRYSTALS_TO_WIN = 20
    MAX_MOVES = 50
    MAX_STEPS = 2000 # Safety limit

    # --- Colors ---
    COLOR_BG = (15, 20, 45)
    CRYSTAL_COLORS = [
        (255, 50, 50),   # Red
        (50, 255, 50),   # Green
        (80, 80, 255),   # Blue
    ]
    SHADOW_COLORS = [
        (128, 25, 25),
        (25, 128, 25),
        (40, 40, 128),
    ]
    HIGHLIGHT_COLORS = [
        (255, 150, 150),
        (150, 255, 150),
        (180, 180, 255),
    ]
    COLOR_GRID = (30, 40, 70)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_PARTICLE = (255, 255, 200)

    # --- Visuals ---
    TILE_WIDTH = 54
    TILE_HEIGHT = TILE_WIDTH * 0.5
    CRYSTAL_Z_HEIGHT = 20
    ORIGIN_X = 640 // 2
    ORIGIN_Y = 100

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.grid = None
        self.cursor_pos = None
        self.moves_remaining = None
        self.score = None
        self.crystals_cleared = None
        self.game_over = None
        self.win_state = None
        self.steps = None
        self.last_turn_reward = None
        
        self.animation_queue = deque()
        self.particles = []
        self.is_animating = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_remaining = self.MAX_MOVES
        self.crystals_cleared = 0
        self.game_over = False
        self.win_state = None # None, "win", "lose"
        self.last_turn_reward = 0

        self.cursor_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        self._generate_initial_grid()

        self.animation_queue.clear()
        self.particles.clear()
        self.is_animating = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        turn_reward = 0
        terminated = False
        
        # --- Animation Update ---
        self._update_animations()
        
        # Process new action only if no animations are running
        if not self.is_animating and not self.game_over:
            movement, _, shift_held = action[0], action[1] == 1, action[2] == 1

            if movement != 0: # Any directional input
                if shift_held:
                    # --- Cursor Movement ---
                    dy, dx = self._get_delta_from_movement(movement)
                    self.cursor_pos[0] = max(0, min(self.GRID_HEIGHT - 1, self.cursor_pos[0] + dy))
                    self.cursor_pos[1] = max(0, min(self.GRID_WIDTH - 1, self.cursor_pos[1] + dx))
                else:
                    # --- Crystal Swap Attempt ---
                    self.moves_remaining -= 1
                    turn_reward, terminated = self._process_swap(movement)
                    self.last_turn_reward = turn_reward
                    self.is_animating = len(self.animation_queue) > 0

        # Check for game over conditions after an action is fully resolved
        if not self.is_animating and not self.game_over:
            if self.crystals_cleared >= self.CRYSTALS_TO_WIN:
                terminated = True
                self.game_over = True
                self.win_state = "win"
                turn_reward += 100
            elif self.moves_remaining <= 0:
                terminated = True
                self.game_over = True
                self.win_state = "lose"
                turn_reward -= 10
        
        # Safety termination
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        self.score += turn_reward

        return (
            self._get_observation(),
            turn_reward,
            terminated,
            False,
            self._get_info(),
        )
        
    def _process_swap(self, movement):
        """Processes a swap action, checks for matches, and handles chain reactions."""
        y1, x1 = self.cursor_pos
        dy, dx = self._get_delta_from_movement(movement)
        y2, x2 = y1 + dy, x1 + dx
        
        # Check if swap is within bounds
        if not (0 <= y2 < self.GRID_HEIGHT and 0 <= x2 < self.GRID_WIDTH):
            return -0.2, False # Penalty for invalid move attempt

        # Perform swap
        self._add_animation("swap", pos1=(y1, x1), pos2=(y2, x2), duration=15)
        self.grid[y1, x1], self.grid[y2, x2] = self.grid[y2, x2], self.grid[y1, x1]
        
        total_reward = 0
        chain_level = 0
        
        while True:
            matches = self._find_matches()
            if not matches:
                break
                
            # Reward for matches
            if chain_level > 0:
                total_reward += 5 # Chain reaction bonus
            total_reward += len(matches)
            self.crystals_cleared += len(matches)
            
            # Animate and remove matched crystals
            for y, x in matches:
                self._add_animation("destroy", pos=(y, x), duration=15, delay=15)
                # sfx: crystal_break.wav
                self._create_particles(y, x)
                self.grid[y, x] = -1 # Mark for removal
            
            # Apply gravity and refill
            self._apply_gravity_and_refill()
            chain_level += 1
        
        if chain_level == 0: # No matches found from the initial swap
            return -0.2, False # Penalty for a non-productive move
        
        return total_reward, False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render grid lines
        for r in range(self.GRID_HEIGHT + 1):
            p1 = self._iso_to_screen(r, 0)
            p2 = self._iso_to_screen(r, self.GRID_WIDTH)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, (p1[0], p1[1]+self.TILE_HEIGHT/2), (p2[0], p2[1]+self.TILE_HEIGHT/2))
        for c in range(self.GRID_WIDTH + 1):
            p1 = self._iso_to_screen(0, c)
            p2 = self._iso_to_screen(self.GRID_HEIGHT, c)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, (p1[0], p1[1]+self.TILE_HEIGHT/2), (p2[0], p2[1]+self.TILE_HEIGHT/2))

        # Render particles
        self._render_particles()

        # Render crystals
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                crystal_type = self.grid[r, c]
                if crystal_type != -1:
                    self._draw_crystal(r, c, crystal_type)
        
        # Render cursor
        if not self.game_over:
            self._draw_cursor()

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Moves
        moves_text = self.font_main.render(f"Moves: {self.moves_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (620 - moves_text.get_width(), 10))

        # Crystals Cleared
        cleared_text = self.font_small.render(f"Crystals Cleared: {self.crystals_cleared} / {self.CRYSTALS_TO_WIN}", True, self.COLOR_UI_TEXT)
        self.screen.blit(cleared_text, (20, 370))
        
        # Last Reward
        if self.last_turn_reward != 0:
            reward_str = f"+{self.last_turn_reward}" if self.last_turn_reward > 0 else f"{self.last_turn_reward}"
            reward_color = (100, 255, 100) if self.last_turn_reward > 0 else (255, 100, 100)
            reward_text = self.font_small.render(reward_str, True, reward_color)
            self.screen.blit(reward_text, (20, 45))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            msg = "YOU WIN!" if self.win_state == "win" else "GAME OVER"
            color = (100, 255, 100) if self.win_state == "win" else (255, 100, 100)
            
            end_text = self.font_main.render(msg, True, color)
            text_rect = end_text.get_rect(center=(320, 200))
            
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "crystals_cleared": self.crystals_cleared,
        }
        
    # --- Helper Functions ---
    
    def _get_delta_from_movement(self, movement):
        if movement == 1: return -1, 0  # Up
        if movement == 2: return 1, 0   # Down
        if movement == 3: return 0, -1  # Left
        if movement == 4: return 0, 1   # Right
        return 0, 0

    def _iso_to_screen(self, r, c):
        x = self.ORIGIN_X + (c - r) * self.TILE_WIDTH / 2
        y = self.ORIGIN_Y + (c + r) * self.TILE_HEIGHT / 2
        return int(x), int(y)

    def _generate_initial_grid(self):
        self.grid = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        while True:
            matches = self._find_matches()
            if not matches:
                break
            for r, c in matches:
                self.grid[r, c] = self.np_random.integers(0, self.NUM_COLORS)

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color = self.grid[r, c]
                if color == -1: continue
                # Horizontal
                if c < self.GRID_WIDTH - 2 and self.grid[r, c+1] == color and self.grid[r, c+2] == color:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                # Vertical
                if r < self.GRID_HEIGHT - 2 and self.grid[r+1, c] == color and self.grid[r+2, c] == color:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _apply_gravity_and_refill(self):
        fall_delay = 30
        for c in range(self.GRID_WIDTH):
            empty_count = 0
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] == -1:
                    empty_count += 1
                elif empty_count > 0:
                    crystal_type = self.grid[r, c]
                    self.grid[r + empty_count, c] = crystal_type
                    self.grid[r, c] = -1
                    self._add_animation("fall", pos=(r, c), end_pos=(r + empty_count, c), 
                                      crystal_type=crystal_type, duration=15, delay=fall_delay)
            # Refill
            for i in range(empty_count):
                new_crystal = self.np_random.integers(0, self.NUM_COLORS)
                self.grid[i, c] = new_crystal
                self._add_animation("fall", pos=(-(empty_count - i), c), end_pos=(i, c),
                                  crystal_type=new_crystal, duration=15, delay=fall_delay)

    def _draw_crystal(self, r, c, crystal_type):
        pos = (r, c)
        scale = 1.0
        alpha = 255
        
        # Check for animations affecting this crystal
        for anim in self.animation_queue:
            p = anim['progress']
            if anim['type'] == 'swap':
                if anim['pos1'] == (r, c):
                    pos = self._interpolate_pos(anim['pos1'], anim['pos2'], p)
                elif anim['pos2'] == (r, c):
                    pos = self._interpolate_pos(anim['pos2'], anim['pos1'], p)
            elif anim['type'] == 'fall' and anim['crystal_type'] == crystal_type and anim['end_pos'] == (r,c):
                pos = self._interpolate_pos(anim['pos'], anim['end_pos'], p)
            elif anim['type'] == 'destroy' and anim['pos'] == (r, c):
                scale = 1.0 - p
                alpha = 255 * (1.0 - p)

        screen_x, screen_y = self._iso_to_screen(pos[0], pos[1])
        
        w = (self.TILE_WIDTH / 2) * scale
        h = (self.TILE_HEIGHT / 2) * scale
        z = self.CRYSTAL_Z_HEIGHT * scale
        
        if w < 1 or h < 1: return
        
        main_color = self.CRYSTAL_COLORS[crystal_type]
        shadow_color = self.SHADOW_COLORS[crystal_type]
        highlight_color = self.HIGHLIGHT_COLORS[crystal_type]
        
        points = {
            'top': (screen_x, screen_y - h),
            'right': (screen_x + w, screen_y),
            'bottom': (screen_x, screen_y + h),
            'left': (screen_x - w, screen_y),
            'top_3d': (screen_x, screen_y - h - z),
            'right_3d': (screen_x + w, screen_y - z),
            'bottom_3d': (screen_x, screen_y + h - z),
            'left_3d': (screen_x - w, screen_y - z),
        }
        
        # Draw with gfxdraw for antialiasing and alpha
        pygame.gfxdraw.filled_polygon(self.screen, [points['left'], points['bottom'], points['right'], points['top']], (*shadow_color, int(alpha)))
        pygame.gfxdraw.filled_polygon(self.screen, [points['left_3d'], points['left'], points['bottom'], points['bottom_3d']], (*shadow_color, int(alpha)))
        pygame.gfxdraw.filled_polygon(self.screen, [points['right_3d'], points['right'], points['bottom'], points['bottom_3d']], (*main_color, int(alpha)))
        pygame.gfxdraw.filled_polygon(self.screen, [points['top_3d'], points['left_3d'], points['right_3d']], (*highlight_color, int(alpha)))
        
    def _draw_cursor(self):
        y, x = self.cursor_pos
        screen_x, screen_y = self._iso_to_screen(y, x)
        w, h = self.TILE_WIDTH / 2, self.TILE_HEIGHT / 2
        
        points = [
            (screen_x, screen_y - h),
            (screen_x + w, screen_y),
            (screen_x, screen_y + h),
            (screen_x - w, screen_y),
        ]
        
        # Pulsing effect
        pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2
        color = tuple(int(c1 * pulse + c2 * (1 - pulse)) for c1, c2 in zip(self.COLOR_CURSOR, (255, 255, 255)))
        
        pygame.draw.lines(self.screen, color, True, points, 3)

    def _create_particles(self, r, c):
        screen_x, screen_y = self._iso_to_screen(r, c)
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = self.np_random.integers(20, 40 + 1)
            self.particles.append([screen_x, screen_y, vx, vy, lifetime])

    def _render_particles(self):
        for p in self.particles[:]:
            p[0] += p[2] # x += vx
            p[1] += p[3] # y += vy
            p[4] -= 1    # lifetime--
            
            if p[4] <= 0:
                self.particles.remove(p)
            else:
                size = int(p[4] / 8)
                if size > 0:
                    alpha = int(255 * (p[4] / 40))
                    color = (*self.COLOR_PARTICLE, alpha)
                    pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), size, color)

    def _add_animation(self, anim_type, duration, **kwargs):
        self.animation_queue.append({
            "type": anim_type,
            "duration": duration,
            "timer": 0,
            "progress": 0.0,
            **kwargs
        })

    def _update_animations(self):
        if not self.animation_queue:
            self.is_animating = False
            return
            
        # Update only the first animation in the sequence until its delay is over
        current_anim = self.animation_queue[0]
        
        delay = current_anim.get('delay', 0)
        if delay > 0:
            current_anim['delay'] -= 1
            return
            
        current_anim['timer'] += 1
        current_anim['progress'] = min(1.0, current_anim['timer'] / current_anim['duration'])
        
        if current_anim['progress'] >= 1.0:
            self.animation_queue.popleft()
            if not self.animation_queue:
                self.is_animating = False

    def _interpolate_pos(self, pos1, pos2, progress):
        y1, x1 = pos1
        y2, x2 = pos2
        interp_y = y1 + (y2 - y1) * progress
        interp_x = x1 + (x2 - x1) * progress
        return interp_y, interp_x

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

# To run and visualize the game (optional)
if __name__ == '__main__':
    # This block will fail if run directly because the environment is configured
    # for headless operation (SDL_VIDEODRIVER="dummy").
    # To visualize, you would need to modify the __init__ method to not set the
    # dummy driver and use pygame.display.set_mode.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # The following code is for interactive visualization and will raise an error
    # in the default headless configuration.
    try:
        screen = pygame.display.set_mode((640, 400))
        pygame.display.set_caption("Crystal Cavern")
        clock = pygame.time.Clock()
        
        running = True
        while running:
            action = [0, 0, 0] # no-op
            keys = pygame.key.get_pressed()
            
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
            
            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                    if event.key == pygame.K_ESCAPE:
                        running = False

            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_remaining']}")
            
            if terminated:
                print("Game Over!")

            # Display the observation from the environment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(30) # Run at 30 FPS

    except pygame.error as e:
        print(f"Caught Pygame error, likely due to headless mode: {e}")
        print("To run interactively, please modify the environment's __init__ method.")

    pygame.quit()