import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:08:58.605835
# Source Brief: brief_02133.md
# Brief Index: 2133
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple

# Helper class for Cards
Card = namedtuple('Card', ['type', 'value', 'name'])

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Explore Julia set fractals by applying algorithm and color cards. Discover and save unique artistic "
        "creations to complete your gallery."
    )
    user_guide = (
        "Controls: ↑/↓ to zoom, ←/→ to rotate the fractal. Press space to apply the current card and shift to "
        "save a unique creation to your gallery."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    UI_WIDTH = 240
    FRACTAL_WIDTH, FRACTAL_HEIGHT = SCREEN_WIDTH - UI_WIDTH, SCREEN_HEIGHT

    # Performance optimization: render fractal at a lower resolution
    FRACTAL_RENDER_SCALE = 2
    FRACTAL_RENDER_WIDTH = FRACTAL_WIDTH // FRACTAL_RENDER_SCALE
    FRACTAL_RENDER_HEIGHT = FRACTAL_HEIGHT // FRACTAL_RENDER_SCALE
    MAX_ITERATIONS = 50

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_UI_BG = (25, 25, 40)
    COLOR_UI_BORDER = (80, 80, 120)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_DIM = (150, 150, 170)
    COLOR_CARD_BG = (40, 40, 60)
    COLOR_CARD_BORDER = (100, 100, 140)
    COLOR_TYPE_ALGO = (255, 150, 50)
    COLOR_TYPE_COLOR = (50, 150, 255)
    COLOR_GALLERY_EMPTY = (50, 50, 70)
    COLOR_GALLERY_SUCCESS = (100, 255, 100)

    # Game rules
    MAX_STEPS = 2000
    GALLERY_SIZE = 12
    UNIQUENESS_THRESHOLD = 0.15 # Avg pixel difference for a fractal to be "unique"

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

        # Fonts
        try:
            self.FONT_UI = pygame.font.SysFont("Consolas", 16)
            self.FONT_CARD = pygame.font.SysFont("Consolas", 14)
            self.FONT_TITLE = pygame.font.SysFont("Consolas", 18, bold=True)
        except pygame.error:
            self.FONT_UI = pygame.font.SysFont(None, 20)
            self.FONT_CARD = pygame.font.SysFont(None, 18)
            self.FONT_TITLE = pygame.font.SysFont(None, 24, bold=True)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.deck = []
        self.current_card = None
        self.active_palette = []
        self.active_julia_c = 0j
        self.zoom = 1.0
        self.rotation = 0.0
        self.gallery_thumbnails = []
        self.gallery_hashes = []
        self.was_space_held = False
        self.was_shift_held = False
        self.fractal_surface = pygame.Surface((self.FRACTAL_RENDER_WIDTH, self.FRACTAL_RENDER_HEIGHT))
        self.fractal_params_changed = True
        self.last_reward_info = ""

        # Initialize state
        # self.reset() # reset is called by the environment runner
        # self.validate_implementation() # this is for dev, not needed in final code


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.zoom = 1.5
        self.rotation = 0.0
        
        self.gallery_thumbnails = []
        self.gallery_hashes = []
        
        self.was_space_held = False
        self.was_shift_held = False

        self._create_deck()
        self._apply_card(self._draw_card()) # Start with a random palette
        self._apply_card(self._draw_card()) # And a random algorithm
        self._draw_card() # Draw the first playable card
        
        self.fractal_params_changed = True
        self.last_reward_info = "Game Reset"

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0
        self.last_reward_info = ""
        self.fractal_params_changed = False

        movement, space_action, shift_action = action[0], action[1], action[2]
        space_pressed = space_action == 1 and not self.was_space_held
        shift_pressed = shift_action == 1 and not self.was_shift_held

        # --- Handle Actions ---
        if movement == 1: # Zoom In
            self.zoom *= 1.05
            self.fractal_params_changed = True
        elif movement == 2: # Zoom Out
            self.zoom /= 1.05
            self.fractal_params_changed = True
        elif movement == 3: # Rotate CCW
            self.rotation -= 0.05
            self.fractal_params_changed = True
        elif movement == 4: # Rotate CW
            self.rotation += 0.05
            self.fractal_params_changed = True
        
        # Clamp values to prevent instability
        self.zoom = max(0.1, min(self.zoom, 10.0))
        self.rotation %= (2 * math.pi)

        if space_pressed: # Apply Card
            # SFX: Card apply sound
            self._apply_card(self.current_card)
            self._draw_card()
            self.fractal_params_changed = True
            self.last_reward_info = "Card Applied"
        
        if shift_pressed: # Add to Gallery
            if self._add_to_gallery():
                # SFX: Success sound
                reward += 1.0
                self.score += 1
                self.last_reward_info = "+1.0: Art Added!"
            else:
                # SFX: Failure sound
                reward -= 0.1 
                self.last_reward_info = "-0.1: Not Unique"

        # Continuous reward for creating a unique fractal
        if self._is_current_fractal_unique():
            reward += 0.1
            if not self.last_reward_info: # Don't overwrite more important messages
                 self.last_reward_info = "+0.1: Unique Fractal"
        
        # --- Termination ---
        terminated = False
        if len(self.gallery_thumbnails) >= self.GALLERY_SIZE:
            reward += 100.0
            self.score += 100
            terminated = True
            self.game_over = True
            self.last_reward_info = "+100: Gallery Complete!"
            # SFX: Victory fanfare
        elif self.steps >= self.MAX_STEPS:
            reward -= 100.0
            self.score -= 100
            terminated = True
            self.game_over = True
            self.last_reward_info = "-100: Time Out"
            # SFX: Game over sound

        # Update button-press state trackers
        self.was_space_held = space_action == 1
        self.was_shift_held = shift_action == 1

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
            "gallery_size": len(self.gallery_thumbnails),
            "zoom": self.zoom,
            "rotation": self.rotation,
        }

    # --- Game Logic Helpers ---

    def _create_deck(self):
        self.deck = []
        # Create color cards
        for _ in range(10):
            palette = self._generate_palette()
            name = f"Palette #{random.randint(100,999)}"
            self.deck.append(Card('COLOR', palette, name))
        # Create algorithm cards
        for _ in range(10):
            c = complex(random.uniform(-1, 1), random.uniform(-1, 1))
            name = f"Julia C={c.real:.2f}{c.imag:+.2f}i"
            self.deck.append(Card('ALGORITHM', c, name))
        random.shuffle(self.deck)

    def _generate_palette(self):
        palette = []
        num_colors = self.MAX_ITERATIONS
        c1 = [random.uniform(0, 255) for _ in range(3)]
        c2 = [random.uniform(0, 255) for _ in range(3)]
        c3 = [random.uniform(0, 255) for _ in range(3)]
        for i in range(num_colors):
            t = i / (num_colors -1)
            if t < 0.5:
                t2 = t * 2
                r = int(c1[0] * (1 - t2) + c2[0] * t2)
                g = int(c1[1] * (1 - t2) + c2[1] * t2)
                b = int(c1[2] * (1 - t2) + c2[2] * t2)
            else:
                t2 = (t - 0.5) * 2
                r = int(c2[0] * (1 - t2) + c3[0] * t2)
                g = int(c2[1] * (1 - t2) + c3[1] * t2)
                b = int(c2[2] * (1 - t2) + c3[2] * t2)
            palette.append((r,g,b))
        return palette

    def _draw_card(self):
        if not self.deck:
            self._create_deck()
        self.current_card = self.deck.pop(0)
        return self.current_card

    def _apply_card(self, card):
        if card.type == 'COLOR':
            self.active_palette = card.value
        elif card.type == 'ALGORITHM':
            self.active_julia_c = card.value
        self.fractal_params_changed = True

    def _add_to_gallery(self):
        if len(self.gallery_thumbnails) >= self.GALLERY_SIZE:
            return False
        
        current_hash = self._get_fractal_hash()
        is_unique = self._is_hash_unique(current_hash)
        
        if is_unique:
            thumbnail = pygame.transform.smoothscale(self.fractal_surface, (50, 50))
            self.gallery_thumbnails.append(thumbnail)
            self.gallery_hashes.append(current_hash)
            return True
        return False

    def _get_fractal_hash(self):
        """Generates a simplified hash of the current fractal for uniqueness checks."""
        # Downscale further for a coarse comparison
        small_surf = pygame.transform.smoothscale(self.fractal_surface, (16, 16))
        return pygame.surfarray.array3d(small_surf).tobytes()

    def _is_hash_unique(self, new_hash):
        return new_hash not in self.gallery_hashes

    def _is_current_fractal_unique(self):
        if not self.gallery_hashes:
            return True
        return self._is_hash_unique(self._get_fractal_hash())

    # --- Rendering Methods ---

    def _render_game(self):
        if self.fractal_params_changed:
            self._render_fractal_to_surface()
        
        scaled_fractal = pygame.transform.scale(self.fractal_surface, (self.FRACTAL_WIDTH, self.FRACTAL_HEIGHT))
        self.screen.blit(scaled_fractal, (0, 0))

    def _render_fractal_to_surface(self):
        w, h = self.FRACTAL_RENDER_WIDTH, self.FRACTAL_RENDER_HEIGHT
        
        # Create a grid of complex numbers
        re, im = np.meshgrid(np.linspace(-2, 2, w), np.linspace(-2, 2, h))
        c_grid = (re + 1j * im) / self.zoom

        # Apply rotation
        rot_rad = self.rotation
        c_grid *= np.exp(-1j * rot_rad)
        
        z = c_grid.copy()
        c = self.active_julia_c
        
        iterations = np.zeros(z.shape, dtype=int)
        mask = np.ones(z.shape, dtype=bool)

        for i in range(self.MAX_ITERATIONS):
            z[mask] = z[mask]**2 + c
            diverged = np.abs(z) > 2
            iterations[mask & diverged] = i
            mask[diverged] = False
        
        # Map iterations to colors
        color_array = np.array(self.active_palette, dtype=np.uint8)
        pixels = color_array[iterations]
        
        pygame.surfarray.blit_array(self.fractal_surface, pixels.swapaxes(0, 1))
        self.fractal_params_changed = False


    def _render_ui(self):
        ui_x = self.FRACTAL_WIDTH
        # Panel background
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (ui_x, 0, self.UI_WIDTH, self.SCREEN_HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_UI_BORDER, (ui_x, 0), (ui_x, self.SCREEN_HEIGHT), 2)
        
        # --- Gallery ---
        self._draw_text("Gallery", self.FONT_TITLE, ui_x + 15, 10)
        for i in range(self.GALLERY_SIZE):
            row, col = i // 4, i % 4
            gx = ui_x + 20 + col * 55
            gy = 40 + row * 55
            if i < len(self.gallery_thumbnails):
                self.screen.blit(self.gallery_thumbnails[i], (gx, gy))
                pygame.gfxdraw.rectangle(self.screen, (gx, gy, 50, 50), self.COLOR_GALLERY_SUCCESS)
            else:
                pygame.gfxdraw.rectangle(self.screen, (gx, gy, 50, 50), self.COLOR_GALLERY_EMPTY)

        # --- Current Card ---
        card_y = 220
        self._draw_text("Current Card", self.FONT_TITLE, ui_x + 15, card_y)
        card_rect = pygame.Rect(ui_x + 15, card_y + 30, self.UI_WIDTH - 30, 80)
        pygame.draw.rect(self.screen, self.COLOR_CARD_BG, card_rect)
        
        if self.current_card:
            card_color = self.COLOR_TYPE_ALGO if self.current_card.type == 'ALGORITHM' else self.COLOR_TYPE_COLOR
            pygame.draw.rect(self.screen, card_color, (card_rect.x, card_rect.y, card_rect.width, 20))
            self._draw_text(self.current_card.type, self.FONT_CARD, card_rect.x + 5, card_rect.y + 2, color=self.COLOR_BG)
            
            # Word wrap for card name
            words = self.current_card.name.split(' ')
            lines = []
            current_line = ""
            for word in words:
                if self.FONT_CARD.size(current_line + word)[0] < card_rect.width - 10:
                    current_line += word + " "
                else:
                    lines.append(current_line)
                    current_line = word + " "
            lines.append(current_line)
            
            for i, line in enumerate(lines):
                 self._draw_text(line, self.FONT_CARD, card_rect.x + 5, card_rect.y + 25 + i * 15, color=self.COLOR_TEXT)
        pygame.gfxdraw.rectangle(self.screen, card_rect, self.COLOR_CARD_BORDER)


        # --- Info & Controls ---
        info_y = 340
        self._draw_text(f"Score: {self.score}", self.FONT_UI, ui_x + 15, info_y)
        self._draw_text(f"Steps: {self.steps}/{self.MAX_STEPS}", self.FONT_UI, ui_x + 15, info_y + 20)
        self._draw_text(f"Reward: {self.last_reward_info}", self.FONT_CARD, 10, self.SCREEN_HEIGHT - 20, color=self.COLOR_TEXT_DIM)

        # Control hints
        controls_y = self.SCREEN_HEIGHT - 60
        self._draw_text("[SPACE] Apply Card", self.FONT_UI, ui_x + 15, controls_y, color=self.COLOR_TEXT_DIM)
        self._draw_text("[SHIFT] Add to Gallery", self.FONT_UI, ui_x + 15, controls_y + 20, color=self.COLOR_TEXT_DIM)

    def _draw_text(self, text, font, x, y, color=COLOR_TEXT):
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, (int(x), int(y)))

    def close(self):
        pygame.quit()

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # To run with display, you might need to unset the dummy video driver
    # e.g., by commenting out the `os.environ` line at the top
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        print("Running in headless mode. No display will be shown.")
        print("Comment out 'os.environ.setdefault(\"SDL_VIDEODRIVER\", \"dummy\")' to run with a display.")
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a Pygame window to display the environment
    try:
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Fractal Artist Gym Environment")
        display_on = True
    except pygame.error:
        print("Pygame display could not be initialized. Running without visual feedback.")
        display_on = False

    running = True
    while running:
        # --- Human Controls ---
        movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
        space = 0 # 0=released, 1=held
        shift = 0 # 0=released, 1=held
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        # --- Pygame Event Loop ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment...")
                obs, info = env.reset()
        
        # --- Step the Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Gallery: {info['gallery_size']}/{env.GALLERY_SIZE}")

        if terminated or truncated:
            print("Episode finished!")
            print(f"Final Score: {info['score']}")
            obs, info = env.reset() # Auto-reset on termination
            
        # --- Render to Screen ---
        if display_on:
            # The observation is already a rendered image, so we just need to display it
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS

    env.close()