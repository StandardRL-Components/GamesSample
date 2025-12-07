import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to jump 1 space. Space to jump 2 spaces forward. Shift to jump 2 spaces backward."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hop between platforms to reach the top. Collect at least 15 stars to win. Watch your step!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 16
    GRID_ROWS = 10
    CELL_WIDTH = SCREEN_WIDTH // GRID_COLS
    CELL_HEIGHT = SCREEN_HEIGHT // GRID_ROWS
    
    WIN_STARS = 15
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG_TOP = (20, 30, 80)
    COLOR_BG_BOTTOM = (60, 80, 160)
    COLOR_PLATFORM = (60, 120, 220)
    COLOR_PLATFORM_OUTLINE = (40, 80, 180)
    COLOR_PLATFORM_ACTIVE = (120, 200, 255)
    COLOR_PLATFORM_ACTIVE_OUTLINE = (80, 160, 220)
    COLOR_TOP_PLATFORM = (255, 200, 0)
    COLOR_TOP_PLATFORM_OUTLINE = (200, 160, 0)
    
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_OUTLINE = (200, 200, 200)
    COLOR_PLAYER_ACTIVE = (100, 255, 255)  # Added missing attribute
    
    COLOR_STAR = (255, 230, 0)
    COLOR_STAR_OUTLINE = (200, 180, 0)
    
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_SHADOW = (0, 0, 0, 128)

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
        self.font_ui = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.platforms = []
        self.stars = []
        self.particles = []
        self.player_pos = [0, 0]
        self.player_facing_dir = 1
        self.star_count = 0
        self.steps = 0
        self.game_over_state = None # 'win', 'fall', 'timeout'
        self.last_platform_y = 0
        self.top_platform_center_x = 0
        
        # self.reset() # This is called by validate_implementation
        # self.validate_implementation() # Removed from init to avoid issues with external verifiers

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.star_count = 0
        self.game_over_state = None
        self.particles = []
        
        self._generate_level()
        
        start_platform = self.platforms[-1]
        start_x_grid = start_platform['rect'].centerx // self.CELL_WIDTH
        self.player_pos = [start_x_grid, self.GRID_ROWS - 1]
        self.player_facing_dir = 1
        self.last_platform_y = self.player_pos[1]

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.platforms = []
        self.stars = []

        # Bottom platform
        bottom_rect = pygame.Rect(0, (self.GRID_ROWS - 1) * self.CELL_HEIGHT, self.SCREEN_WIDTH, self.CELL_HEIGHT)
        self.platforms.append({'rect': bottom_rect, 'grid_y': self.GRID_ROWS - 1, 'is_top': False})

        # Intermediate platforms
        for y_grid in range(self.GRID_ROWS - 2, 0, -1):
            platforms_in_row_below = [p for p in self.platforms if p['grid_y'] == y_grid + 1]
            if not platforms_in_row_below: continue

            num_platforms = self.np_random.integers(1, 3)
            possible_x_starts = list(range(self.GRID_COLS))
            shuffled_starts = self.np_random.permutation(possible_x_starts)


            for x_start in shuffled_starts:
                if num_platforms <= 0: break
                width = self.np_random.integers(2, 6)
                if x_start + width >= self.GRID_COLS: continue

                is_reachable = False
                for p_below in platforms_in_row_below:
                    min_reach = (p_below['rect'].left // self.CELL_WIDTH) - 2
                    max_reach = (p_below['rect'].right // self.CELL_WIDTH) + 2
                    if max(x_start, min_reach) < min(x_start + width, max_reach):
                        is_reachable = True
                        break
                
                if is_reachable:
                    new_rect = pygame.Rect(x_start * self.CELL_WIDTH, y_grid * self.CELL_HEIGHT, width * self.CELL_WIDTH, self.CELL_HEIGHT)
                    
                    # Prevent overlap
                    is_overlapping = False
                    for p in self.platforms:
                        if p['grid_y'] == y_grid and new_rect.colliderect(p['rect']):
                            is_overlapping = True
                            break
                    if not is_overlapping:
                        self.platforms.append({'rect': new_rect, 'grid_y': y_grid, 'is_top': False})
                        num_platforms -= 1

        # Top platform
        top_width_grid = 8
        top_x_grid = (self.GRID_COLS - top_width_grid) // 2
        top_rect = pygame.Rect(top_x_grid * self.CELL_WIDTH, 0, top_width_grid * self.CELL_WIDTH, self.CELL_HEIGHT)
        self.platforms.insert(0, {'rect': top_rect, 'grid_y': 0, 'is_top': True})
        self.top_platform_center_x = top_x_grid + top_width_grid // 2

        # Generate stars
        for p in self.platforms:
            if p['grid_y'] == self.GRID_ROWS - 1: continue # No stars on bottom platform
            num_stars = self.np_random.integers(0, p['rect'].width // self.CELL_WIDTH)
            for _ in range(num_stars):
                star_x_grid = p['rect'].left // self.CELL_WIDTH + self.np_random.integers(0, p['rect'].width // self.CELL_WIDTH)
                star_y_grid = p['grid_y']
                if not any(s['pos'] == [star_x_grid, star_y_grid] for s in self.stars):
                    self.stars.append({'pos': [star_x_grid, star_y_grid]})

    def step(self, action):
        if self.game_over_state:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        terminated = False
        
        self._update_particles()

        old_pos = list(self.player_pos)
        old_platform = self._get_platform_at_grid(old_pos[0], old_pos[1])
        
        target_pos = list(old_pos)
        jumped = True
        
        if space_pressed:
            target_pos[0] += 2 * self.player_facing_dir
        elif shift_pressed:
            target_pos[0] -= 2 * self.player_facing_dir
        elif movement != 0:
            if movement == 1: # Up
                target_platform_obj = self._find_platform_relative(old_pos[0], old_pos[1], 'up')
                if target_platform_obj: target_pos = [old_pos[0], target_platform_obj['grid_y']]
                else: target_pos = [old_pos[0], old_pos[1] - 1] # Fail jump
            elif movement == 2: # Down
                target_platform_obj = self._find_platform_relative(old_pos[0], old_pos[1], 'down')
                if target_platform_obj: target_pos = [old_pos[0], target_platform_obj['grid_y']]
                else: target_pos = [old_pos[0], old_pos[1] + 1] # Fail jump
            elif movement == 3: # Left
                target_pos[0] -= 1
                self.player_facing_dir = -1
            elif movement == 4: # Right
                target_pos[0] += 1
                self.player_facing_dir = 1
        else: # No-op
            jumped = False

        target_platform = self._get_platform_at_grid(target_pos[0], target_pos[1])

        if jumped and not target_platform:
            # Fall
            terminated = True
            reward = -100.0
            self.game_over_state = 'fall'
            self._create_particles(self._grid_to_pixel(old_pos), 30, (200, 200, 255), 'fall')
        else:
            # Successful jump or no-op
            if jumped:
                self._create_particles(self._grid_to_pixel(old_pos), 10, self.COLOR_PLAYER, 'jump')
                self.player_pos = target_pos
                self._create_particles(self._grid_to_pixel(self.player_pos), 20, self.COLOR_PLAYER_ACTIVE, 'land')
                # SFX: Jump, Land

            # Star collection
            collected_star = self._collect_star()
            if collected_star:
                reward += 1.0
                self.star_count += 1
                self._create_particles(self._grid_to_pixel(self.player_pos), 30, self.COLOR_STAR, 'star')
                # SFX: Star collect

            new_platform = self._get_platform_at_grid(self.player_pos[0], self.player_pos[1])
            new_platform_y = new_platform['grid_y']
            
            # Reward for vertical movement
            if jumped:
                if new_platform_y < self.last_platform_y:
                    reward += 5.0
                elif new_platform_y > self.last_platform_y:
                    reward -= 0.5
            
            # Reward for horizontal progress
            old_dist = abs(old_pos[0] - self.top_platform_center_x)
            new_dist = abs(self.player_pos[0] - self.top_platform_center_x)
            if new_dist < old_dist:
                reward += 0.1 * (old_dist - new_dist)

            self.last_platform_y = new_platform_y
            
            # Win condition
            if new_platform.get('is_top') and self.star_count >= self.WIN_STARS:
                terminated = True
                reward += 100.0
                self.game_over_state = 'win'
                # SFX: Win
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            self.game_over_state = 'timeout'
            reward -= 10.0 # Small penalty for timeout
        
        return self._get_observation(), float(reward), terminated, False, self._get_info()

    def _get_platform_at_grid(self, grid_x, grid_y):
        if not (0 <= grid_x < self.GRID_COLS and 0 <= grid_y < self.GRID_ROWS):
            return None
        pixel_x = grid_x * self.CELL_WIDTH + self.CELL_WIDTH / 2
        pixel_y = grid_y * self.CELL_HEIGHT + self.CELL_HEIGHT / 2
        for p in self.platforms:
            if p['rect'].collidepoint(pixel_x, pixel_y):
                return p
        return None

    def _find_platform_relative(self, grid_x, grid_y, direction):
        if direction == 'up':
            relevant_platforms = sorted([p for p in self.platforms if p['grid_y'] < grid_y], key=lambda p: p['grid_y'], reverse=True)
        else: # down
            relevant_platforms = sorted([p for p in self.platforms if p['grid_y'] > grid_y], key=lambda p: p['grid_y'])
        
        for p in relevant_platforms:
            if p['rect'].left <= grid_x * self.CELL_WIDTH < p['rect'].right:
                return p
        return None

    def _collect_star(self):
        for i, star in enumerate(self.stars):
            if star['pos'] == self.player_pos:
                self.stars.pop(i)
                return True
        return False

    def _get_observation(self):
        self._render_background()
        self._render_platforms()
        self._render_stars()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.star_count,
            "steps": self.steps,
            "player_pos": list(self.player_pos),
            "player_facing": self.player_facing_dir,
        }

    def _grid_to_pixel(self, grid_pos):
        x = grid_pos[0] * self.CELL_WIDTH + self.CELL_WIDTH // 2
        y = grid_pos[1] * self.CELL_HEIGHT + self.CELL_HEIGHT // 2
        return [x, y]

    # --- Rendering Methods ---
    
    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_platforms(self):
        current_platform = self._get_platform_at_grid(self.player_pos[0], self.player_pos[1])
        for p in self.platforms:
            rect = p['rect'].copy()
            rect.height -= 4 # Visual gap
            rect.top += 2
            
            is_active = (current_platform == p)
            is_top = p.get('is_top', False)
            
            if is_top:
                color = self.COLOR_TOP_PLATFORM
                outline_color = self.COLOR_TOP_PLATFORM_OUTLINE
            elif is_active:
                color = self.COLOR_PLATFORM_ACTIVE
                outline_color = self.COLOR_PLATFORM_ACTIVE_OUTLINE
            else:
                color = self.COLOR_PLATFORM
                outline_color = self.COLOR_PLATFORM_OUTLINE

            pygame.gfxdraw.box(self.screen, rect, color)
            pygame.gfxdraw.rectangle(self.screen, rect, outline_color)

    def _render_stars(self):
        for star in self.stars:
            pos = self._grid_to_pixel(star['pos'])
            radius = self.CELL_WIDTH // 4
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, self.COLOR_STAR)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), radius, self.COLOR_STAR_OUTLINE)

    def _render_player(self):
        if self.game_over_state == 'fall': return
        
        pos = self._grid_to_pixel(self.player_pos)
        size = self.CELL_WIDTH * 0.6
        rect = pygame.Rect(pos[0] - size / 2, pos[1] - size / 2, size, size)
        
        pygame.gfxdraw.box(self.screen, rect, self.COLOR_PLAYER)
        pygame.gfxdraw.rectangle(self.screen, rect, self.COLOR_PLAYER_OUTLINE)

    def _render_ui(self):
        # Star count
        star_text = f"★ {self.star_count} / {self.WIN_STARS}"
        self._draw_text(star_text, (20, 20), self.font_ui, self.COLOR_UI_TEXT)
        
        # Height
        height = (self.GRID_ROWS - 1) - self.player_pos[1]
        height_text = f"Height: {height}"
        self._draw_text(height_text, (self.SCREEN_WIDTH - 150, 20), self.font_ui, self.COLOR_UI_TEXT)

        if self.game_over_state:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            
            if self.game_over_state == 'win':
                msg = "YOU WIN!"
                color = self.COLOR_STAR
            elif self.game_over_state == 'fall':
                msg = "GAME OVER"
                color = self.COLOR_PLATFORM_ACTIVE
            else: # timeout
                msg = "TIME'S UP"
                color = (200, 100, 100)

            self._draw_text(msg, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 30), self.font_ui, color, center=True)
            self._draw_text("Press any action to reset", (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 20), self.font_small, self.COLOR_UI_TEXT, center=True)


    def _draw_text(self, text, pos, font, color, center=False):
        shadow_surface = font.render(text, True, self.COLOR_UI_SHADOW)
        text_surface = font.render(text, True, color)
        
        if center:
            text_rect = text_surface.get_rect(center=pos)
        else:
            text_rect = text_surface.get_rect(topleft=pos)
        
        shadow_rect = text_rect.copy()
        shadow_rect.x += 2
        shadow_rect.y += 2
        
        self.screen.blit(shadow_surface, shadow_rect)
        self.screen.blit(text_surface, text_rect)

    # --- Particle System ---

    def _create_particles(self, pos, count, color, p_type):
        for _ in range(count):
            if p_type == 'land':
                vel = [(self.np_random.random() - 0.5) * 4, -self.np_random.random() * 2]
            elif p_type == 'star':
                angle = self.np_random.random() * 2 * math.pi
                speed = self.np_random.random() * 3 + 1
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            elif p_type == 'fall':
                vel = [(self.np_random.random() - 0.5) * 2, (self.np_random.random() - 0.5) * 2]
            else: # jump
                vel = [(self.np_random.random() - 0.5) * 2, self.np_random.random() * 2]
            
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'lifespan': self.np_random.integers(15, 30),
                'color': color,
                'size': self.np_random.random() * 3 + 2
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['lifespan'] -= 1
            p['size'] -= 0.05
        self.particles = [p for p in self.particles if p['lifespan'] > 0 and p['size'] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 20))))
            color = (*p['color'], alpha)
            size = int(p['size'])
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (size, size), size)
                self.screen.blit(s, (int(p['pos'][0] - size), int(p['pos'][1] - size)))
    
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Beginning validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        self.reset()
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
    # This block allows you to play the game directly
    # For human play, we want to see the screen.
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv(render_mode="rgb_array")
    
    # Run validation
    try:
        env.validate_implementation()
    except Exception as e:
        print(f"Validation failed: {e}")
        exit()

    obs, info = env.reset()
    terminated = False
    
    # Pygame setup for human play
    pygame.display.set_caption("Platform Hopper")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # [movement, space, shift]
        
        # Event handling
        do_step = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                
                if terminated:
                    obs, info = env.reset()
                    terminated = False
                    continue
                
                do_step = True

        if do_step:
            # Map keys to actions
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            if keys[pygame.K_SPACE]:
                action[1] = 1
            
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1

            # Since auto_advance is False, we only step on a key press
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Stars: {info['score']}, Terminated: {terminated}")
        
        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for human play

    pygame.quit()