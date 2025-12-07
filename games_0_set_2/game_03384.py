
# Generated: 2025-08-27T23:12:28.033019
# Source Brief: brief_03384.md
# Brief Index: 3384

        
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


# --- Constants ---
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
WORLD_WIDTH, WORLD_HEIGHT = 4000, 800
FPS = 30
PLAYER_ACCEL = 0.8
PLAYER_FRICTION = 0.85
MAX_SPEED = 6
GRAVITY = 0.8
JUMP_STRENGTH = -15
CRYSTALS_TO_WIN = 5
MAX_STEPS = 1000

# --- Colors ---
COLOR_BG = (20, 25, 40)
COLOR_PLAYER = (255, 255, 255)
COLOR_PLATFORM = (100, 110, 130)
COLOR_PITFALL = (0, 0, 0)
COLOR_UI_TEXT = (220, 220, 240)
CRYSTAL_COLORS = [
    (255, 80, 80),   # Red
    (80, 255, 80),   # Green
    (80, 120, 255),  # Blue
    (255, 255, 80),  # Yellow
    (200, 80, 255),  # Purple
]


class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(random.uniform(-3, 3), random.uniform(-5, 1))
        self.color = color
        self.lifespan = random.randint(20, 40)
        self.radius = random.randint(3, 6)

    def update(self):
        self.pos += self.vel
        self.vel.y += 0.1  # Particle gravity
        self.lifespan -= 1
        self.radius = max(0, self.radius - 0.1)

    def draw(self, surface, camera_offset):
        if self.lifespan > 0 and self.radius > 0:
            pos_on_screen = self.pos - camera_offset
            pygame.draw.circle(
                surface, self.color,
                (int(pos_on_screen.x), int(pos_on_screen.y)),
                int(self.radius)
            )


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Collect all 5 crystals to win."
    )

    game_description = (
        "Explore a procedurally generated crystal cavern. "
        "Collect all the crystals while avoiding the dark pitfalls."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        # Initialize state variables
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.platforms = []
        self.crystals = []
        self.pitfalls = []
        self.particles = []
        self.background_stars = []
        self.camera_offset = pygame.Vector2(0, 0)
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.reset()
        self.validate_implementation()

    def _generate_level(self):
        """Procedurally generates the cavern layout."""
        self.platforms.clear()
        self.crystals.clear()
        self.pitfalls.clear()
        self.background_stars.clear()

        # World boundaries
        self.platforms.append(pygame.Rect(-10, 0, 10, WORLD_HEIGHT)) # Left wall
        self.platforms.append(pygame.Rect(WORLD_WIDTH, 0, 10, WORLD_HEIGHT)) # Right wall

        for _ in range(200):
            x = self.np_random.integers(0, WORLD_WIDTH)
            y = self.np_random.integers(0, WORLD_HEIGHT)
            layer = self.np_random.choice([0.2, 0.4, 0.6])
            self.background_stars.append(((x, y), layer))

        start_x = WORLD_WIDTH / 2
        start_y = WORLD_HEIGHT / 2
        start_platform = pygame.Rect(start_x - 100, start_y, 200, 40)
        self.platforms.append(start_platform)
        self.player_pos = pygame.Vector2(start_platform.centerx, start_platform.top - 30)

        last_platform = start_platform
        platforms_to_generate = 30
        for _ in range(platforms_to_generate):
            w = self.np_random.integers(80, 201)
            h = self.np_random.integers(20, 41)
            dx = self.np_random.integers(int(w/2) + 40, int(w/2) + 151) * self.np_random.choice([-1, 1])
            dy = self.np_random.integers(-100, 101)
            
            new_x = last_platform.centerx + dx
            new_y = last_platform.centery + dy
            
            new_x = max(w/2, min(WORLD_WIDTH - w/2, new_x))
            new_y = max(100, min(WORLD_HEIGHT - h - 100, new_y))

            new_platform = pygame.Rect(new_x - w/2, new_y - h/2, w, h)
            
            if not any(new_platform.colliderect(p) for p in self.platforms):
                self.platforms.append(new_platform)
                last_platform = new_platform

        safe_platforms = [p for p in self.platforms if p.width > 50 and p.y > 100 and p.y < WORLD_HEIGHT - 150]
        if len(safe_platforms) < CRYSTALS_TO_WIN:
             return self._generate_level() # Retry if not enough platforms
        
        crystal_indices = self.np_random.choice(len(safe_platforms), size=CRYSTALS_TO_WIN, replace=False)
        crystal_platforms = [safe_platforms[i] for i in crystal_indices]

        for i, p in enumerate(crystal_platforms):
            crystal_pos = (p.centerx, p.top - 20)
            crystal_rect = pygame.Rect(crystal_pos[0] - 10, crystal_pos[1] - 10, 20, 20)
            self.crystals.append((crystal_rect, CRYSTAL_COLORS[i % len(CRYSTAL_COLORS)]))

        self.pitfalls.append(pygame.Rect(0, WORLD_HEIGHT - 10, WORLD_WIDTH, 20))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_level()
        
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.particles.clear()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        # --- Physics and Player Update ---
        if movement == 3: # Left
            self.player_vel.x -= PLAYER_ACCEL
        elif movement == 4: # Right
            self.player_vel.x += PLAYER_ACCEL
        
        self.player_vel.x *= PLAYER_FRICTION
        if abs(self.player_vel.x) < 0.1: self.player_vel.x = 0
        self.player_vel.x = max(-MAX_SPEED, min(MAX_SPEED, self.player_vel.x))
        
        self.player_vel.y += GRAVITY
        self.player_vel.y = min(20, self.player_vel.y) # Terminal velocity

        if movement == 1 and self.on_ground:
            self.player_vel.y = JUMP_STRENGTH
            self.on_ground = False
            # sfx: jump

        # --- Collision Detection ---
        self.on_ground = False
        player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 20, 20, 30)
        
        self.player_pos.x += self.player_vel.x
        player_rect.x = int(self.player_pos.x - 10)
        for platform in self.platforms:
            if player_rect.colliderect(platform):
                if self.player_vel.x > 0: player_rect.right = platform.left
                elif self.player_vel.x < 0: player_rect.left = platform.right
                self.player_pos.x = player_rect.centerx
                self.player_vel.x = 0

        self.player_pos.y += self.player_vel.y
        player_rect.y = int(self.player_pos.y - 20)
        for platform in self.platforms:
            if player_rect.colliderect(platform):
                if self.player_vel.y > 0:
                    player_rect.bottom = platform.top
                    self.on_ground = True
                    self.player_vel.y = 0
                elif self.player_vel.y < 0:
                    player_rect.top = platform.bottom
                    self.player_vel.y = 0
                self.player_pos.y = player_rect.centery

        # --- Update Game State ---
        reward = -0.01 # Time penalty
        player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 20, 20, 30)

        for crystal_rect, color in self.crystals[:]:
            if player_rect.colliderect(crystal_rect):
                self.crystals.remove((crystal_rect, color))
                self.score += 1
                reward += 10
                # sfx: crystal_collect
                for _ in range(30):
                    self.particles.append(Particle(crystal_rect.centerx, crystal_rect.centery, color))

        for pit in self.pitfalls:
            if player_rect.colliderect(pit):
                self.game_over = True
                reward = -10
                # sfx: fall_death
                break
        
        for p in self.particles[:]:
            p.update()
            if p.lifespan <= 0: self.particles.remove(p)

        # --- Termination Checks ---
        terminated = self.game_over
        if self.score >= CRYSTALS_TO_WIN:
            reward += 100
            terminated = True
            # sfx: win_game
        
        self.steps += 1
        if self.steps >= MAX_STEPS:
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _render_game(self):
        self.camera_offset.x = self.player_pos.x - SCREEN_WIDTH / 2
        self.camera_offset.y = self.player_pos.y - SCREEN_HEIGHT / 2
        self.camera_offset.x = max(0, min(self.camera_offset.x, WORLD_WIDTH - SCREEN_WIDTH))
        self.camera_offset.y = max(0, min(self.camera_offset.y, WORLD_HEIGHT - SCREEN_HEIGHT))

        for pos, layer in self.background_stars:
            screen_x = (pos[0] - self.camera_offset.x * layer) % SCREEN_WIDTH
            screen_y = (pos[1] - self.camera_offset.y * layer) % SCREEN_HEIGHT
            size = max(1, int(2 * (1 - layer)))
            color_val = int(150 * (1-layer) + 50)
            pygame.draw.rect(self.screen, (color_val, color_val, color_val), (screen_x, screen_y, size, size))

        render_area = pygame.Rect(self.camera_offset.x, self.camera_offset.y, SCREEN_WIDTH, SCREEN_HEIGHT)
        for p in self.platforms:
            if p.colliderect(render_area):
                pygame.draw.rect(self.screen, COLOR_PLATFORM, p.move(-self.camera_offset.x, -self.camera_offset.y))
        
        for p in self.pitfalls:
            if p.colliderect(render_area):
                pygame.draw.rect(self.screen, COLOR_PITFALL, p.move(-self.camera_offset.x, -self.camera_offset.y))

        for c_rect, c_color in self.crystals:
            if c_rect.colliderect(render_area):
                pos_on_screen = c_rect.center - self.camera_offset
                pulse = math.sin(self.steps * 0.1) * 3 + 12
                glow_color = tuple(min(255, int(c*0.5)) for c in c_color)
                pygame.gfxdraw.filled_circle(self.screen, int(pos_on_screen.x), int(pos_on_screen.y), int(pulse), glow_color)
                
                points = [
                    (pos_on_screen.x, pos_on_screen.y - 10), (pos_on_screen.x + 8, pos_on_screen.y),
                    (pos_on_screen.x, pos_on_screen.y + 10), (pos_on_screen.x - 8, pos_on_screen.y)
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, c_color)
                pygame.gfxdraw.filled_polygon(self.screen, points, c_color)

        for p in self.particles: p.draw(self.screen, self.camera_offset)

        player_screen_pos = self.player_pos - self.camera_offset
        player_rect_draw = pygame.Rect(int(player_screen_pos.x) - 10, int(player_screen_pos.y) - 20, 20, 30)
        pygame.draw.rect(self.screen, COLOR_PLAYER, player_rect_draw)

    def _render_ui(self):
        score_text = f"Crystals: {self.score} / {CRYSTALS_TO_WIN}"
        score_surf = self.font_large.render(score_text, True, COLOR_UI_TEXT)
        self.screen.blit(score_surf, (20, 10))

        step_text = f"Steps: {self.steps} / {MAX_STEPS}"
        step_surf = self.font_small.render(step_text, True, COLOR_UI_TEXT)
        self.screen.blit(step_surf, (SCREEN_WIDTH - step_surf.get_width() - 20, 15))

    def _get_observation(self):
        self.screen.fill(COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

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
        
        # print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Crystal Caverns")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    terminated = False
    running = True

    while running:
        # --- Event Handling ---
        movement = 0 # No-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_UP]:
            movement = 1
        
        if keys[pygame.K_r]: # Reset key
             obs, info = env.reset()
             terminated = False

        # --- Action and Step ---
        action = [movement, 0, 0] # Space/Shift not used
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
            terminated = False

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to get it onto the display screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(FPS)

    env.close()