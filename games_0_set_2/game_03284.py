import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ← and → to select a potion. Press Space to add it to the cauldron. Match the target color before time runs out!"
    )

    game_description = (
        "A fast-paced isometric puzzle game. Race against the clock to mix potions and perfectly match a target color."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.FPS = 30

        # Game Constants
        self.GAME_DURATION_SECONDS = 60
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS
        self.WIN_THRESHOLD = 15  # Max RGB distance for a win
        self.MOVE_COOLDOWN_FRAMES = 5

        # Visuals & Fonts
        self._init_colors()
        self._init_fonts()
        
        # Game state variables (initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.win_state = None
        self.time_left = None
        self.selected_potion_index = None
        self.prev_space_held = None
        self.last_move_time = None
        self.potion_colors = None
        self.current_color = None
        self.target_color = None
        self.last_color_distance = None
        self.particles = None
        self.mix_animation_timer = None
        self.np_random = None

        # self.reset() is called by the wrapper, but we can call it here for standalone use
        # self.validate_implementation() is also for dev; not needed in final version

    def _init_colors(self):
        self.COLOR_BG = (28, 32, 36)
        self.COLOR_UI_TEXT = (220, 220, 230)
        self.COLOR_UI_ACCENT = (255, 190, 0)
        self.COLOR_UI_DANGER = (255, 80, 80)
        self.COLOR_CAULDRON_OUTER = (60, 65, 70)
        self.COLOR_CAULDRON_INNER = (40, 45, 50)
        self.COLOR_CAULDRON_RIM = (80, 85, 90)
        self.POTION_COLOR_POOL = [
            (255, 60, 60),    # Red
            (60, 255, 60),    # Green
            (60, 120, 255),   # Blue
            (255, 255, 60),   # Yellow
            (255, 60, 255),   # Magenta
            (60, 255, 255),   # Cyan
        ]

    def _init_fonts(self):
        self.font_huge = pygame.font.Font(None, 80)
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = None
        self.time_left = self.GAME_DURATION_SECONDS

        self.selected_potion_index = 0
        self.prev_space_held = False
        self.last_move_time = -self.MOVE_COOLDOWN_FRAMES

        pool_indices = self.np_random.choice(len(self.POTION_COLOR_POOL), 4, replace=False)
        self.potion_colors = [self.POTION_COLOR_POOL[i] for i in pool_indices]
        
        self.current_color = np.array([50, 45, 40], dtype=float)  # Murky brown start
        self.target_color = self._generate_target_color()
        self.last_color_distance = self._color_distance(self.current_color, self.target_color)

        self.particles = []
        self.mix_animation_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_left -= 1 / self.FPS
        reward = 0

        self._handle_input(action)
        self._update_animations()
        
        current_dist = self._color_distance(self.current_color, self.target_color)
        
        # Continuous reward for progress
        if current_dist < self.last_color_distance:
            reward += 0.1
        elif current_dist > self.last_color_distance:
            reward -= 0.1
        self.last_color_distance = current_dist

        # Check for termination conditions
        terminated = False
        if current_dist < self.WIN_THRESHOLD:
            reward += 100
            self.game_over = True
            terminated = True
            self.win_state = 'win'
        elif self.time_left <= 0 or self.steps >= self.MAX_STEPS:
            reward -= 10
            self.game_over = True
            terminated = True
            self.win_state = 'loss'

        self.score += reward
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True # Per Gymnasium API, if truncated, terminated should also be true.
            
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        if self.steps > self.last_move_time + self.MOVE_COOLDOWN_FRAMES:
            if movement == 3:  # Left
                self.selected_potion_index = max(0, self.selected_potion_index - 1)
                self.last_move_time = self.steps
            elif movement == 4:  # Right
                self.selected_potion_index = min(3, self.selected_potion_index + 1)
                self.last_move_time = self.steps

        if space_held and not self.prev_space_held:
            self._mix_potion()
        self.prev_space_held = space_held

    def _mix_potion(self):
        potion_to_add = np.array(self.potion_colors[self.selected_potion_index], dtype=float)
        self.current_color = self.current_color * 0.8 + potion_to_add * 0.2
        self.current_color = np.clip(self.current_color, 0, 255)
        
        self.mix_animation_timer = self.FPS // 2  # 0.5 second animation
        self._spawn_particles(self.potion_colors[self.selected_potion_index])

    def _spawn_particles(self, color):
        cauldron_center_x = self.WIDTH // 2
        cauldron_center_y = self.HEIGHT // 2 + 30
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed - self.np_random.uniform(1, 3) # Upward bias
            lifetime = self.np_random.integers(20, 40)
            size = self.np_random.uniform(2, 5)
            self.particles.append([cauldron_center_x, cauldron_center_y, vx, vy, size, lifetime, color])

    def _update_animations(self):
        if self.mix_animation_timer > 0:
            self.mix_animation_timer -= 1
        
        # Update particles
        new_particles = []
        for p in self.particles:
            p[0] += p[2]  # x += vx
            p[1] += p[3]  # y += vy
            p[3] += 0.1   # Gravity
            p[4] -= 0.1   # Shrink
            p[5] -= 1     # Lifetime
            if p[4] > 0 and p[5] > 0:
                new_particles.append(p)
        self.particles = new_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw potions
        potion_y = self.HEIGHT - 50
        for i in range(4):
            potion_x = self.WIDTH // 2 - 180 + i * 120
            is_selected = (i == self.selected_potion_index)
            self._draw_potion(self.screen, (potion_x, potion_y), self.potion_colors[i], is_selected)

        # Draw cauldron
        self._draw_cauldron(self.screen, (self.WIDTH // 2, self.HEIGHT // 2 + 30))
        
        # Draw particles
        for x, y, _, _, size, lifetime, color in self.particles:
            alpha = max(0, min(255, int(255 * (lifetime / 20))))
            s = max(0, int(size))
            if s > 0:
                temp_surf = pygame.Surface((s*2, s*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*color, alpha), (s, s), s)
                self.screen.blit(temp_surf, (int(x) - s, int(y) - s), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Draw Target Color Swatch
        pygame.draw.rect(self.screen, self.target_color.astype(int), (self.WIDTH - 130, 50, 100, 100))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (self.WIDTH - 130, 50, 100, 100), 2)
        self._draw_text("TARGET", self.font_small, self.COLOR_UI_TEXT, self.WIDTH - 80, 35)

        # Draw Timer
        timer_color = self.COLOR_UI_DANGER if self.time_left < 10 else self.COLOR_UI_ACCENT
        time_str = f"{max(0, self.time_left):.1f}"
        self._draw_text(time_str, self.font_large, timer_color, self.WIDTH // 2, 30)

        # Draw Score
        self._draw_text(f"Score: {self.score:.1f}", self.font_medium, self.COLOR_UI_TEXT, 10, 10, align="topleft")
        
        # Draw Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            if self.win_state == 'win':
                self._draw_text("MATCH!", self.font_huge, (100, 255, 100), self.WIDTH / 2, self.HEIGHT / 2 - 20)
            else:
                self._draw_text("TIME'S UP!", self.font_huge, self.COLOR_UI_DANGER, self.WIDTH / 2, self.HEIGHT / 2 - 20)

    def _draw_text(self, text, font, color, x, y, align="center"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "center":
            text_rect.center = (int(x), int(y))
        elif align == "topleft":
            text_rect.topleft = (int(x), int(y))
        self.screen.blit(text_surface, text_rect)

    def _draw_iso_poly(self, surf, color, points):
        # Lighten color for anti-aliasing to blend better with dark background
        light_color = tuple(min(255, c + 30) for c in color)
        pygame.gfxdraw.aapolygon(surf, points, light_color)
        pygame.gfxdraw.filled_polygon(surf, points, color)

    def _project(self, x, y, z, origin):
        ox, oy = origin
        iso_x = ox + (x - y)
        iso_y = oy + (x + y) / 2 - z
        return int(iso_x), int(iso_y)

    def _draw_potion(self, surf, pos, color, selected):
        x, y = pos
        w, d, h = 20, 20, 40 # bottle dimensions
        
        # Pulsing selection highlight
        if selected:
            pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
            size = 40 + pulse * 10
            alpha = 50 + pulse * 50
            highlight_surf = pygame.Surface((size*2, size), pygame.SRCALPHA)
            pygame.draw.ellipse(highlight_surf, (*self.COLOR_UI_ACCENT, alpha), (0,0,size*2,size))
            surf.blit(highlight_surf, (x - size, y - size/2 + d))

        # Bottle glass (top and front)
        top_color = tuple(min(255, c + 60) for c in color)
        side_color = tuple(min(255, c + 30) for c in color)
        dark_side_color = color

        # Top ellipse
        pygame.draw.ellipse(surf, top_color, (x - w, y - d/2, w*2, d))
        pygame.draw.ellipse(surf, tuple(min(255, c+80) for c in top_color), (x - w, y - d/2, w*2, d), 2)
        
        # Body
        body_rect = pygame.Rect(x - w, y, w * 2, h)
        pygame.draw.rect(surf, side_color, body_rect)
        pygame.draw.line(surf, dark_side_color, (x, y), (x, y + h), 2) # Center line
        pygame.draw.line(surf, top_color, (x - w, y), (x - w, y + h), 2)
        pygame.draw.line(surf, dark_side_color, (x + w, y), (x + w, y + h), 2)

        # Base ellipse
        pygame.draw.ellipse(surf, dark_side_color, (x - w, y + h - d/2, w*2, d))

    def _draw_cauldron(self, surf, pos):
        ox, oy = pos
        w, d, h = 80, 80, 50 # cauldron dimensions

        # Liquid surface
        liquid_color = tuple(int(c) for c in self.current_color)
        liquid_h_offset = 15
        if self.mix_animation_timer > 0:
            wobble = math.sin(self.mix_animation_timer * 0.8) * 3
            liquid_h_offset += wobble
        
        # Back rim
        pygame.draw.arc(surf, self.COLOR_CAULDRON_RIM, (ox - w - 10, oy - d/2 - 10, (w+10)*2, (d+10)*2), math.pi, 2*math.pi, 8)
        
        # Liquid ellipse
        pygame.draw.ellipse(surf, liquid_color, (ox - w, oy - d/2 + liquid_h_offset, w*2, d))
        
        # Front body
        body_rect = pygame.Rect(ox - w, oy + liquid_h_offset, w*2, h)
        # The following line was fixed. pygame.draw.ellipse does not take draw_bottom_left/right arguments.
        # The intended effect (a rounded bottom) is achieved by drawing a full ellipse and then
        # drawing a rectangle over its top half.
        pygame.draw.ellipse(surf, self.COLOR_CAULDRON_OUTER, body_rect, 0)
        pygame.draw.rect(surf, self.COLOR_CAULDRON_OUTER, (ox-w, oy+liquid_h_offset, w*2, h/2))

        # Front rim
        pygame.draw.arc(surf, self.COLOR_CAULDRON_RIM, (ox - w - 10, oy - d/2 - 10, (w+10)*2, (d+10)*2), 0, math.pi, 8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "target_color": self.target_color,
            "current_color": self.current_color,
            "color_distance": self.last_color_distance,
        }

    def _generate_target_color(self):
        temp_color = np.copy(self.current_color)
        # Use self.np_random which is seeded
        num_mixes = self.np_random.integers(2, 4)
        for _ in range(num_mixes):
            potion_index = self.np_random.integers(len(self.potion_colors))
            potion_to_add = np.array(self.potion_colors[potion_index], dtype=float)
            temp_color = temp_color * 0.8 + potion_to_add * 0.2
            temp_color = np.clip(temp_color, 0, 255)
        return temp_color

    def _color_distance(self, c1, c2):
        return np.linalg.norm(np.array(c1) - np.array(c2))

    def render(self):
        return self._get_observation()

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It's a useful way to test and debug your environment
    
    # Un-set the dummy driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # Set up Pygame window for human play
    pygame.display.set_caption("Potion Panic")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print("      POTION PANIC")
    print("="*30)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        # Get keyboard input
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle game exit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if terminated or truncated:
            print(f"Episode finished! Final Score: {info['score']:.2f} in {info['steps']} steps.")
            # Wait for a moment, then reset
            pygame.time.wait(2000)
            obs, info = env.reset(seed=np.random.randint(0, 10000))
            total_reward = 0

        # Cap the frame rate
        clock.tick(env.FPS)
        
    env.close()