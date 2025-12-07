import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:42:09.940792
# Source Brief: brief_01147.md
# Brief Index: 1147
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = (
        "Leap across procedurally generated chromosome platforms, collect genetic letters, and form genes to gain abilities on your journey to the telomere."
    )
    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Collect letters and press space to form a gene."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_LENGTH = 10000  # Total horizontal distance of the level
        self.MAX_STEPS = 2500

        # --- Colors ---
        self.COLOR_BG = (10, 0, 20)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_GLOW = (50, 255, 150, 50)
        self.COLOR_PLATFORM_A = (80, 80, 220)
        self.COLOR_PLATFORM_B = (150, 80, 220)
        self.COLOR_TELOMERE = (255, 255, 100)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.LETTER_COLORS = {
            'A': (255, 80, 80), 'T': (255, 255, 80),
            'C': (255, 160, 80), 'G': (80, 255, 80)
        }

        # --- Physics & Gameplay ---
        self.GRAVITY = 0.6
        self.PLAYER_SPEED = 5
        self.JUMP_STRENGTH = -12
        self.INVENTORY_MAX_SIZE = 7

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # --- Gene Definitions ---
        # Gene sequence -> function to call
        self.gene_recipes = {
            ('G', 'A', 'T'): self._gene_effect_create_platform,
            ('A', 'T', 'G'): self._gene_effect_boost_jump,
            ('C', 'A', 'G'): self._gene_effect_gain_points,
        }
        
        # --- State Variables (initialized in reset) ---
        self.player_pos = None
        self.player_vel = None
        self.on_ground = None
        self.platforms = None
        self.letters = None
        self.particles = None
        self.inventory = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.last_platform_x = None
        self.camera_x = None
        self.last_space_press = None
        self.checkpoints = None
        self.checkpoints_reached = None
        self.difficulty_gap = None
        self.jump_boost_timer = None

        # self.reset() is called by the environment wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = pygame.Vector2(100, 200)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        
        self.platforms = []
        self.letters = []
        self.particles = []
        self.inventory = deque(maxlen=self.INVENTORY_MAX_SIZE)
        
        self.last_platform_x = 0
        self.camera_x = 0
        self.last_space_press = False
        self.difficulty_gap = 100
        self.jump_boost_timer = 0
        
        # --- Checkpoints ---
        num_checkpoints = 4
        self.checkpoints = [
            (i / num_checkpoints) * self.WORLD_LENGTH for i in range(1, num_checkpoints + 1)
        ]
        self.checkpoints_reached = [False] * num_checkpoints

        # --- Procedural Generation ---
        # Create a starting platform
        self.platforms.append(pygame.Rect(20, 250, 200, 20))
        self.last_platform_x = 220
        # Generate the initial screen
        while self.last_platform_x < self.WIDTH + 100:
            self._procedurally_generate_chunk()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- 1. Handle Input ---
        space_press = space_held and not self.last_space_press
        self.last_space_press = space_held
        
        # Player movement
        if movement == 3:  # Left
            self.player_vel.x = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_vel.x = self.PLAYER_SPEED
        else:
            self.player_vel.x = 0
        
        # Player jump
        if movement == 1 and self.on_ground:
            jump_power = self.JUMP_STRENGTH * 1.5 if self.jump_boost_timer > 0 else self.JUMP_STRENGTH
            self.player_vel.y = jump_power
            self.on_ground = False
            # sfx: jump.wav
            self._create_particle_burst(self.player_pos + pygame.Vector2(0, 15), 5, (200, 200, 255))

        # Form gene
        if space_press:
            gene_reward = self._form_gene()
            reward += gene_reward

        # --- 2. Update Game Logic ---
        self.steps += 1
        self.difficulty_gap += 0.05 / 500  # Gradual difficulty increase
        if self.jump_boost_timer > 0:
            self.jump_boost_timer -= 1

        # Update player physics
        self.player_vel.y += self.GRAVITY
        self.player_pos += self.player_vel
        self.on_ground = False

        # --- 3. Handle Collisions ---
        player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 15, 20, 30)
        
        # Platform collisions
        for plat in self.platforms:
            if player_rect.colliderect(plat) and self.player_vel.y > 0:
                # Check if player was above the platform in the previous frame
                if (player_rect.bottom - self.player_vel.y) <= plat.top:
                    self.player_pos.y = plat.top - 15
                    player_rect.bottom = plat.top
                    self.player_vel.y = 0
                    self.on_ground = True
                    # sfx: land.wav

        # Letter collection
        for letter in self.letters[:]:
            if player_rect.colliderect(letter['rect']):
                self.inventory.append(letter['type'])
                self.letters.remove(letter)
                reward += 0.1
                self.score += 1
                # sfx: collect.wav
                self._create_particle_burst(letter['rect'].center, 10, self.LETTER_COLORS[letter['type']])

        # --- 4. Update World ---
        self._update_world()

        # --- 5. Check Termination Conditions ---
        terminated = False
        truncated = False
        
        # Fall failure
        if self.player_pos.y > self.HEIGHT + 50:
            self.game_over = True
            terminated = True
            reward -= 100
            # sfx: fall.wav

        # Telomere victory
        telomere_rect = pygame.Rect(self.WORLD_LENGTH, 0, 50, self.HEIGHT)
        if player_rect.colliderect(telomere_rect):
            self.game_over = True
            terminated = True
            reward += 100
            self.score += 1000
            # sfx: win.wav
        
        # Max steps
        if self.steps >= self.MAX_STEPS:
            truncated = True
            terminated = True

        # Checkpoint rewards
        for i, cp_x in enumerate(self.checkpoints):
            if not self.checkpoints_reached[i] and self.player_pos.x > cp_x:
                self.checkpoints_reached[i] = True
                reward += 50
                self.score += 500
                # sfx: checkpoint.wav
                self._create_particle_burst((self.player_pos.x, self.player_pos.y - 50), 50, (255, 255, 100))

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _procedurally_generate_chunk(self):
        gap = self.np_random.uniform(self.difficulty_gap * 0.8, self.difficulty_gap * 1.2)
        new_plat_x = self.last_platform_x + gap
        
        last_y = self.platforms[-1].y if self.platforms else self.HEIGHT / 2
        new_plat_y = self.np_random.uniform(
            max(150, last_y - 80), min(self.HEIGHT - 50, last_y + 80)
        )
        
        new_plat_width = self.np_random.uniform(80, 250)
        
        new_platform = pygame.Rect(new_plat_x, new_plat_y, new_plat_width, 20)
        self.platforms.append(new_platform)
        
        # Chance to spawn a letter on the new platform
        if self.np_random.random() < 0.7:
            letter_type = self.np_random.choice(list(self.LETTER_COLORS.keys()))
            letter_pos_x = new_plat_x + new_plat_width / 2
            letter_pos_y = new_plat_y - 25
            self.letters.append({
                'type': letter_type,
                'rect': pygame.Rect(letter_pos_x - 10, letter_pos_y - 10, 20, 20),
                'pos': pygame.Vector2(letter_pos_x, letter_pos_y)
            })
            
        self.last_platform_x = new_plat_x + new_plat_width

    def _update_world(self):
        # Smooth camera follow with lead
        target_camera_x = self.player_pos.x - self.WIDTH / 3
        self.camera_x += (target_camera_x - self.camera_x) * 0.1

        # Keep player within horizontal bounds
        self.player_pos.x = max(self.player_pos.x, self.camera_x + 10)
        self.player_pos.x = min(self.player_pos.x, self.camera_x + self.WIDTH - 10)

        # Generate new chunks if needed
        if self.camera_x + self.WIDTH > self.last_platform_x - 200:
            self._procedurally_generate_chunk()

        # Prune off-screen objects
        self.platforms = [p for p in self.platforms if p.right > self.camera_x]
        self.letters = [l for l in self.letters if l['rect'].right > self.camera_x]
        
        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _form_gene(self):
        inventory_tuple = tuple(self.inventory)
        for gene, effect_func in self.gene_recipes.items():
            if len(inventory_tuple) >= len(gene) and inventory_tuple[-len(gene):] == gene:
                # Gene found!
                for _ in range(len(gene)):
                    self.inventory.pop()
                
                effect_func()
                # sfx: gene_activate.wav
                self._create_particle_burst(self.player_pos, 30, (255, 255, 255))
                self.score += 100
                return 5.0 # Reward for forming a gene
        return 0.0

    # --- Gene Effects ---
    def _gene_effect_create_platform(self):
        # Creates a temporary platform ahead of the player
        plat_x = self.player_pos.x + 100
        plat_y = self.player_pos.y + 20
        new_platform = pygame.Rect(plat_x, plat_y, 120, 20)
        new_platform.is_temporary = True # Custom attribute
        new_platform.creation_time = self.steps
        self.platforms.append(new_platform)
    
    def _gene_effect_boost_jump(self):
        # Gives the player a temporary jump boost
        self.jump_boost_timer = 30 * 5 # 5 seconds at 30fps
        
    def _gene_effect_gain_points(self):
        # Just a simple score boost
        self.score += 250

    def _get_observation(self):
        # --- Rendering ---
        self.screen.fill(self.COLOR_BG)
        
        # 1. Render Background Elements (Parallax)
        for i in range(15):
            hash_i = (i * 1337) % 1000
            px = (hash_i * 23) % self.WIDTH
            py = (hash_i * 57) % self.HEIGHT
            size = 20 + (hash_i % 40)
            
            # Parallax scrolling effect
            scroll_factor = 0.1 + (hash_i % 5) / 10.0
            display_x = (px - self.camera_x * scroll_factor) % self.WIDTH
            
            color_val = 30 + (hash_i % 20)
            color = (color_val, 0, color_val + 10)
            
            pygame.gfxdraw.filled_circle(self.screen, int(display_x), int(py), int(size), color)

        # 2. Render Game Elements
        self._render_game()
        
        # 3. Render UI
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        cam_x = int(self.camera_x)

        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*p['color'], alpha)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x - cam_x), int(p['pos'].y), size, color)

        # Render Telomere
        telomere_x = self.WORLD_LENGTH - cam_x
        if telomere_x < self.WIDTH + 50:
            for i in range(20):
                angle = (self.steps * 0.02 + i * math.pi * 2 / 20)
                radius = 150 + 20 * math.sin(self.steps * 0.05 + i)
                end_x = telomere_x + radius * math.cos(angle)
                end_y = self.HEIGHT / 2 + radius * math.sin(angle)
                pygame.draw.line(self.screen, self.COLOR_TELOMERE, (telomere_x, self.HEIGHT / 2), (end_x, end_y), 1)
            pygame.gfxdraw.filled_circle(self.screen, int(telomere_x), self.HEIGHT // 2, 50, self.COLOR_TELOMERE)
            pygame.gfxdraw.filled_circle(self.screen, int(telomere_x), self.HEIGHT // 2, 40, (255, 255, 255))

        # Render platforms (chromosomes)
        for plat in self.platforms:
            screen_rect = plat.move(-cam_x, 0)
            self._draw_chromosome_platform(self.screen, screen_rect)
            
            # Fade out temporary platforms
            if hasattr(plat, 'is_temporary'):
                life_left = 300 - (self.steps - plat.creation_time)
                if life_left < 0:
                    if plat in self.platforms: self.platforms.remove(plat)
                elif life_left < 100:
                    alpha = int(255 * (life_left/100))
                    s = pygame.Surface(plat.size, pygame.SRCALPHA)
                    s.fill((0,0,0, 255 - alpha))
                    self.screen.blit(s, screen_rect.topleft)

        # Render letters
        for letter in self.letters:
            pulse = 1 + 0.1 * math.sin(self.steps * 0.1 + letter['pos'].x)
            size = int(10 * pulse)
            color = self.LETTER_COLORS[letter['type']]
            pos = (int(letter['pos'].x - cam_x), int(letter['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, (255, 255, 255))
            
            text_surf = self.font_small.render(letter['type'], True, (0, 0, 0))
            self.screen.blit(text_surf, text_surf.get_rect(center=pos))

        # Render Player
        player_x, player_y = int(self.player_pos.x - cam_x), int(self.player_pos.y)
        
        # Jump boost effect
        if self.jump_boost_timer > 0:
            boost_alpha = 50 + (self.jump_boost_timer % 10) * 10
            pygame.gfxdraw.filled_circle(self.screen, player_x, player_y, 25, (*self.COLOR_PLAYER, boost_alpha))

        # Glow
        glow_size = 25 + 5 * math.sin(self.steps * 0.1)
        pygame.gfxdraw.filled_circle(self.screen, player_x, player_y, int(glow_size), self.COLOR_PLAYER_GLOW)
        
        # Body
        player_rect = pygame.Rect(player_x - 10, player_y - 15, 20, 30)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=10)
        
        # Eyes
        eye_y = player_y - 5
        eye_x_offset = 4 if self.player_vel.x >= 0 else -4
        pygame.draw.circle(self.screen, (0,0,0), (player_x + eye_x_offset, eye_y), 3)

    def _draw_chromosome_platform(self, surface, rect):
        pygame.draw.rect(surface, self.COLOR_PLATFORM_A, rect, border_radius=5)
        num_strands = int(rect.width / 15)
        for i in range(num_strands):
            x = rect.x + i * 15
            y_offset = 8 * math.sin(x * 0.1 + self.steps * 0.05)
            start_pos = (x, rect.centery - y_offset)
            end_pos = (x, rect.centery + y_offset)
            pygame.draw.line(surface, self.COLOR_PLATFORM_B, start_pos, end_pos, 3)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Inventory
        ui_x = 10
        ui_y = 10
        for i, letter_type in enumerate(self.inventory):
            color = self.LETTER_COLORS[letter_type]
            rect = pygame.Rect(ui_x + i * 30, ui_y, 25, 25)
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            
            text_surf = self.font_large.render(letter_type, True, (0,0,0))
            self.screen.blit(text_surf, text_surf.get_rect(center=rect.center))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_x": self.player_pos.x,
            "inventory_size": len(self.inventory),
        }

    def _create_particle_burst(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            life = self.np_random.integers(15, 31)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': life,
                'max_life': life,
                'size': self.np_random.integers(3, 8),
                'color': color,
            })
    
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually.
    # It will not run when the environment is used by the test suite.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use a display driver
    
    env = GameEnv(render_mode="rgb_array")
    
    pygame.display.set_caption("Gene Leap")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    # Action buffer
    action = [0, 0, 0] # [movement, space, shift]

    while not terminated and not truncated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        
        # Movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2 # Not used in game logic
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        else: action[0] = 0
            
        # Space and Shift
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0 # Not used in game logic

        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.metadata["render_fps"])

    env.close()
    print("Game Over!")
    print(f"Final Score: {info['score']}, Total Steps: {info['steps']}")