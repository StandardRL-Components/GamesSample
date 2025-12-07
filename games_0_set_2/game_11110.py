import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:36:44.896778
# Source Brief: brief_01110.md
# Brief Index: 1110
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, pos, vel, radius, color, lifespan):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.radius = radius
        self.color = color
        self.lifespan = lifespan
        self.max_lifespan = lifespan

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        self.radius -= 0.1
        self.vel.y += 0.05 # a little gravity on particles

    def draw(self, surface):
        if self.radius > 0:
            alpha = int(255 * (self.lifespan / self.max_lifespan))
            if alpha > 0:
                pygame.gfxdraw.filled_circle(
                    surface, int(self.pos.x), int(self.pos.y),
                    int(self.radius), (*self.color, alpha)
                )

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A fast-paced platformer. Jump between platforms to collect all the spinning tokens before time runs out."
    )
    user_guide = (
        "Use ←→ arrow keys to move. Press ↑, 'W', or space to jump. Collect all tokens to win."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    MAX_STEPS = 30 * FPS # 30 seconds
    WIN_SCORE = 20

    # Colors
    COLOR_BG = (15, 23, 42)
    COLOR_PLAYER = (250, 204, 21)
    COLOR_PLAYER_GLOW = (234, 179, 8)
    COLOR_PLATFORM = (51, 65, 85)
    COLOR_TOKEN = (34, 197, 94)
    COLOR_TOKEN_GLOW = (74, 222, 128)
    COLOR_TEXT = (241, 245, 249)
    COLOR_SPEED_BAR_BG = (127, 29, 29)
    COLOR_SPEED_BAR_FG = (220, 38, 38)
    
    # Physics
    GRAVITY = 0.35
    PLAYER_JUMP_STRENGTH = -9
    PLAYER_MAX_HSPEED = 5
    SPEED_RECOVERY_RATE = 0.001
    JUMP_SPEED_PENALTY = 0.05

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 16, bold=True)
        
        # State variables are initialized in reset()
        self.level = 1
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_rect = pygame.Rect(0, 0, 24, 24)
        self.speed_multiplier = 1.0
        self.is_grounded = False
        self.jump_just_pressed = False
        self.platforms = []
        self.tokens = []
        self.particles = []
        self.background_stars = []

        # The reset method is called to initialize the state for the first time
        # self.reset() is not called here to avoid duplicate initialization, 
        # as it's standard practice for the user to call reset() before starting.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # On win, level increases. On loss, it resets to 1.
        if self.game_over and self.score < self.WIN_SCORE:
            self.level = 1

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 50)
        self.player_vel = pygame.Vector2(0, 0)
        self.speed_multiplier = 1.0
        self.is_grounded = False
        self.jump_just_pressed = False

        self.particles = []
        self._generate_platforms()
        self.tokens = []
        for _ in range(5): # Keep 5 tokens on screen
            self._spawn_token()
        
        if not self.background_stars:
            for _ in range(100):
                self.background_stars.append(
                    (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT), random.uniform(0.1, 0.5))
                )

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If game is over, do nothing and wait for reset
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_pressed, _ = action
        
        # --- Handle Input ---
        player_x_before = self.player_pos.x
        
        # Horizontal Movement
        if movement == 3:  # Left
            self.player_vel.x = -self.PLAYER_MAX_HSPEED * self.speed_multiplier
        elif movement == 4:  # Right
            self.player_vel.x = self.PLAYER_MAX_HSPEED * self.speed_multiplier
        else:
            self.player_vel.x = 0
        
        # Jump Action
        jump_action_triggered = (movement == 1 or space_pressed == 1)
        if jump_action_triggered and not self.jump_just_pressed and self.is_grounded:
            self.player_vel.y = self.PLAYER_JUMP_STRENGTH
            self.speed_multiplier = max(0.1, self.speed_multiplier - self.JUMP_SPEED_PENALTY)
            self.is_grounded = False
            self.jump_just_pressed = True
            # sfx: jump
            self._create_jump_particles()
        
        if not jump_action_triggered:
            self.jump_just_pressed = False

        # --- Update Game State ---
        self._update_player()
        self._update_particles()
        
        # --- Calculate Reward & Check Termination ---
        reward, collected_token = self._calculate_reward(player_x_before)
        if collected_token:
            self._spawn_token() # Replenish token
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 50 # Win bonus
                self.level += 1
            else:
                reward -= 50 # Timeout penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self):
        # Apply gravity
        self.player_vel.y += self.GRAVITY
        self.player_vel.y = min(self.player_vel.y, 10) # Terminal velocity

        # Update position
        self.player_pos.x += self.player_vel.x
        self.player_pos.y += self.player_vel.y

        # Screen boundaries
        self.player_pos.x = np.clip(self.player_pos.x, 0, self.WIDTH - self.player_rect.width)
        if self.player_pos.y >= self.HEIGHT: # Fell off bottom
            self.player_pos.y = self.HEIGHT
            self.game_over = True # End game if player falls out of bounds

        self.player_rect.topleft = self.player_pos

        # Platform collisions
        self.is_grounded = False
        for plat in self.platforms:
            if (self.player_vel.y > 0 and
                self.player_rect.colliderect(plat) and
                self.player_rect.bottom <= plat.top + self.player_vel.y + 1):
                self.player_pos.y = plat.top - self.player_rect.height
                self.player_vel.y = 0
                self.is_grounded = True
                break
        
        # Speed recovery
        self.speed_multiplier = min(1.0, self.speed_multiplier + self.SPEED_RECOVERY_RATE)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0 and p.radius > 0]
        for p in self.particles:
            p.update()

    def _calculate_reward(self, player_x_before):
        reward = 0
        collected_token = False

        # Token collection reward
        for token in self.tokens[:]:
            token_rect = pygame.Rect(token['pos'].x - 10, token['pos'].y - 10, 20, 20)
            if self.player_rect.colliderect(token_rect):
                self.tokens.remove(token)
                self.score += 1
                reward += 1.0
                collected_token = True
                # sfx: collect_token
                self._create_collect_particles(token['pos'])
                break # Only collect one token per frame
        
        # Movement reward
        if self.tokens:
            # Find nearest token
            player_center = self.player_rect.center
            dists = [pygame.Vector2(player_center).distance_to(t['pos']) for t in self.tokens]
            nearest_token = self.tokens[np.argmin(dists)]
            
            dist_x_before = abs(player_x_before - nearest_token['pos'].x)
            dist_x_after = abs(self.player_pos.x - nearest_token['pos'].x)

            if dist_x_after < dist_x_before:
                reward += 0.1

        return reward, collected_token

    def _check_termination(self):
        return self.steps >= self.MAX_STEPS or self.score >= self.WIN_SCORE or self.game_over

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "speed_multiplier": self.speed_multiplier
        }

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        # Render stars for a parallax effect
        for x, y, speed in self.background_stars:
            # Move stars based on player velocity for parallax
            px = (x - self.player_pos.x * speed) % self.WIDTH
            py = (y - self.player_pos.y * speed * 0.1) % self.HEIGHT
            alpha = int(100 + speed * 155)
            pygame.gfxdraw.pixel(self.screen, int(px), int(py), (*self.COLOR_TEXT, alpha))

    def _render_game(self):
        for p in self.particles:
            p.draw(self.screen)
        
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat, border_radius=3)

        for token in self.tokens:
            self._render_spinning_circle(token['pos'], token['anim_angle'])
            token['anim_angle'] = (token['anim_angle'] + 4) % 360

        self._render_glow_rect(self.player_rect, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)
    
    def _render_ui(self):
        # Token Count
        score_text = self.font_medium.render(f"TOKENS: {self.score}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        timer_text = self.font_large.render(f"{time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH // 2 - timer_text.get_width() // 2, 10))

        # Speed Indicator
        speed_perc_text = self.font_medium.render(f"SPEED: {int(self.speed_multiplier * 100)}%", True, self.COLOR_TEXT)
        self.screen.blit(speed_perc_text, (self.WIDTH - speed_perc_text.get_width() - 10, 10))
        
        bar_width = 150
        bar_height = 10
        bar_x = self.WIDTH - bar_width - 10
        bar_y = 35
        
        fill_width = int(bar_width * self.speed_multiplier)
        pygame.draw.rect(self.screen, self.COLOR_SPEED_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=3)
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_SPEED_BAR_FG, (bar_x, bar_y, fill_width, bar_height), border_radius=3)
    
    def _render_glow_rect(self, rect, color, glow_color):
        glow_rect = rect.inflate(12, 12)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        
        for i in range(6, 0, -1):
            alpha = 80 - i * 12
            pygame.draw.rect(glow_surface, (*glow_color, alpha), (0,0, *glow_rect.size), border_radius=i+3, width=i)
        
        self.screen.blit(glow_surface, glow_rect.topleft)
        pygame.draw.rect(self.screen, color, rect, border_radius=3)
    
    def _render_spinning_circle(self, pos, angle):
        radius = 10
        width_scale = abs(math.cos(math.radians(angle)))
        ellipse_rect = pygame.Rect(0, 0, radius * 2 * width_scale, radius * 2)
        ellipse_rect.center = pos

        glow_rect = ellipse_rect.inflate(10, 10)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surface, (*self.COLOR_TOKEN_GLOW, 80), (0,0, *glow_rect.size))
        self.screen.blit(glow_surface, glow_rect.topleft)

        pygame.draw.ellipse(self.screen, self.COLOR_TOKEN, ellipse_rect)


    def _generate_platforms(self):
        self.platforms = []
        # Ground platform
        self.platforms.append(pygame.Rect(0, self.HEIGHT - 20, self.WIDTH, 20))
        
        num_platforms = 3 + (self.level - 1) * 2
        y = self.HEIGHT - 80
        last_x = self.WIDTH // 2

        for _ in range(num_platforms):
            width = random.randint(80, 150)
            # Ensure platforms are reachable horizontally
            x = last_x + random.randint(-150, 150)
            x = np.clip(x, 0, self.WIDTH - width)
            
            plat_rect = pygame.Rect(x, y, width, 15)
            
            # Prevent overlap
            is_overlapping = any(plat_rect.colliderect(p) for p in self.platforms)
            if not is_overlapping and y > 50:
                self.platforms.append(plat_rect)
                last_x = x
            
            # Move next platform higher
            y -= random.randint(60, 90)

    def _spawn_token(self):
        if not self.platforms: return
        
        # Choose a random platform to spawn a token above
        plat = random.choice(self.platforms)
        
        # Spawn token at a random x on the platform, and a bit above it
        token_x = random.uniform(plat.left + 10, plat.right - 10)
        token_y = plat.top - 30
        
        # Ensure it doesn't spawn inside another platform
        token_rect = pygame.Rect(token_x-10, token_y-10, 20, 20)
        if any(token_rect.colliderect(p) for p in self.platforms):
            # If it's a bad spot, try again
            self._spawn_token()
            return
            
        self.tokens.append({
            'pos': pygame.Vector2(token_x, token_y),
            'anim_angle': random.randint(0, 360)
        })

    def _create_jump_particles(self):
        # sfx: player_jump_whoosh
        for _ in range(10):
            vel_x = random.uniform(-1, 1)
            vel_y = random.uniform(0.5, 2)
            self.particles.append(Particle(
                self.player_rect.midbottom,
                (vel_x, vel_y),
                random.uniform(2, 5),
                self.COLOR_PLAYER,
                random.randint(20, 40)
            ))

    def _create_collect_particles(self, pos):
        # sfx: token_collect_sparkle
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append(Particle(
                pos, vel, random.uniform(3, 6),
                self.COLOR_TOKEN, random.randint(30, 50)
            ))

    def render(self):
        # This method is not used by the gym interface but is useful for human play
        # Re-enable display for human play
        if "dummy" in os.environ.get("SDL_VIDEODRIVER", ""):
            del os.environ["SDL_VIDEODRIVER"]

        pygame.display.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

        self._render_background()
        self._render_game()
        self._render_ui()
        pygame.display.flip()
        self.clock.tick(self.FPS)

    def close(self):
        pygame.quit()


# Example of how to run the environment for human play
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # --- Human Controls ---
    # A/D or Left/Right for movement
    # W, Up Arrow, or Space to Jump
    
    while not terminated:
        # The render method needs to be called to create the display window
        env.render()

        keys = pygame.key.get_pressed()
        
        movement = 0 # None
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            movement = 1 # Up/Jump
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            movement = 3 # Left
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            movement = 4 # Right
            
        space_pressed = 1 if keys[pygame.K_SPACE] else 0
        shift_pressed = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_pressed, shift_pressed]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        if terminated and not (event.type == pygame.QUIT):
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause before resetting
            obs, info = env.reset()
            terminated = False

    env.close()