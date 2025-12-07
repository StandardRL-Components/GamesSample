import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:07:28.516007
# Source Brief: brief_01536.md
# Brief Index: 1536
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# --- Helper Classes for Game Objects ---

class Player:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.w, self.h = 24, 32
        self.vx, self.vy = 0, 0
        self.speed = 3.5
        self.gravity = 0.4
        self.jump_strength = -9
        self.on_ground = False
        self.on_ladder = False
        self.can_jump = True
        self.invincible_timer = 0
        self.speed_boost_timer = 0
        self.color = (50, 150, 255) # Bright Blue

    @property
    def rect(self):
        return pygame.Rect(self.x, self.y, self.w, self.h)

    def update(self, action, space_press, platforms, ladders):
        movement, reward_mod = action, 0
        
        # --- Power-up Timers ---
        if self.invincible_timer > 0: self.invincible_timer -= 1
        if self.speed_boost_timer > 0: self.speed_boost_timer -= 1
        
        current_speed = self.speed * 1.5 if self.speed_boost_timer > 0 else self.speed

        # --- Horizontal Movement ---
        if movement == 3: # Left
            self.vx = -current_speed
            reward_mod -= 0.01
        elif movement == 4: # Right
            self.vx = current_speed
            reward_mod -= 0.01
        else:
            self.vx = 0
        self.x += self.vx

        # --- Ladder Logic ---
        self.on_ladder = any(self.rect.colliderect(ladder.rect) for ladder in ladders)
        if self.on_ladder:
            self.vy = 0
            self.on_ground = True # Can jump from ladders
            if movement == 1: # Up
                self.y -= current_speed
                reward_mod += 0.1
            elif movement == 2: # Down
                self.y += current_speed
        else:
            # --- Gravity ---
            self.vy += self.gravity
            self.on_ground = False

        # --- Jumping ---
        if space_press and self.on_ground and self.can_jump:
            self.vy = self.jump_strength
            self.on_ground = False
            self.can_jump = False
            # sfx: Jump sound

        if not space_press:
            self.can_jump = True

        self.y += self.vy

        # --- Collision Detection (Platforms) ---
        landed_on_platform = None
        for p in platforms:
            if self.rect.colliderect(p.rect) and self.vy > 0 and self.rect.bottom - self.vy <= p.rect.top:
                self.y = p.rect.top - self.h
                self.vy = 0
                self.on_ground = True
                landed_on_platform = p
                break
        
        # --- Screen Bounds ---
        self.x = max(0, min(self.x, 640 - self.w))
        self.y = max(0, min(self.y, 400 - self.h))
        
        return landed_on_platform, reward_mod

    def draw(self, screen, steps):
        # Draw glow effects for power-ups
        if self.invincible_timer > 0:
            alpha = 100 + (math.sin(steps * 0.5) * 50)
            glow_color = (255, 255, 255, max(0, min(255, alpha)))
            self._draw_glow(screen, glow_color, 20)
        elif self.speed_boost_timer > 0:
            alpha = 100 + (math.sin(steps * 0.4) * 50)
            glow_color = (255, 255, 0, max(0, min(255, alpha)))
            self._draw_glow(screen, glow_color, 15)

        pygame.draw.rect(screen, self.color, self.rect)
        
    def _draw_glow(self, screen, color, radius):
        surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, color, (radius, radius), radius)
        screen.blit(surf, (self.rect.centerx - radius, self.rect.centery - radius), special_flags=pygame.BLEND_RGBA_ADD)

class Barrel:
    def __init__(self, x, y, speed_multiplier):
        self.x, self.y = float(x), float(y)
        self.r = 8
        self.vx = (random.choice([-1, 1]) * 2.0) * speed_multiplier
        self.vy = 0
        self.gravity = 0.2
        self.on_ground = False
        self.color = (220, 40, 40) # Bright Red

    @property
    def rect(self):
        return pygame.Rect(self.x - self.r, self.y - self.r, self.r * 2, self.r * 2)

    def update(self, platforms):
        if not self.on_ground:
            self.vy += self.gravity
        else:
            self.x += self.vx
            
        self.y += self.vy
        
        self.on_ground = False
        on_any_platform = False
        for p in platforms:
            if self.rect.colliderect(p.rect) and self.vy >= 0 and self.rect.bottom - self.vy <= p.rect.top:
                self.y = p.rect.top - self.r
                self.vy = 0
                self.on_ground = True
                on_any_platform = True
                break
        
        if on_any_platform:
            current_platform = next((p for p in platforms if p.rect.collidepoint(self.x, self.y + self.r + 1)), None)
            if current_platform:
                if self.x + self.r > current_platform.rect.right or self.x - self.r < current_platform.rect.left:
                    self.vx *= -1
            else: # Fell off an edge
                self.on_ground = False

    def draw(self, screen):
        pygame.gfxdraw.aacircle(screen, int(self.x), int(self.y), self.r, self.color)
        pygame.gfxdraw.filled_circle(screen, int(self.x), int(self.y), self.r, self.color)

class Platform:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)
        self.base_color = (60, 180, 60) # Green

    def draw(self, screen, combo_count, steps):
        color = self.base_color
        if combo_count == 1: color = (255, 255, 0) # Yellow
        elif combo_count == 2: color = (255, 165, 0) # Orange
        elif combo_count >= 3:
            # Flashing white
            if (steps // 4) % 2 == 0:
                color = (255, 255, 255)
            else:
                color = (200, 200, 200)
        pygame.draw.rect(screen, color, self.rect)

class Ladder:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = (100, 100, 110)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect, 2)
        for i in range(self.rect.top, self.rect.bottom, 15):
            pygame.draw.line(screen, self.color, (self.rect.left, i), (self.rect.right, i), 2)

class Particle:
    def __init__(self, x, y, color):
        self.x, self.y = x, y
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-4, 1)
        self.life = random.randint(20, 40)
        self.color = color
        self.radius = self.life / 6

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1
        self.life -= 1
        self.radius = max(0, self.life / 6)

    def draw(self, screen):
        if self.radius > 0:
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), int(self.radius))

# --- Main Gymnasium Environment ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Climb a series of platforms to reach the top while dodging or stomping on falling barrels. "
        "Chain platform landings to unlock power-ups and score big!"
    )
    user_guide = (
        "Use ←→ arrow keys to move and ↑↓ to climb ladders. Press space to jump. "
        "Stomp on barrels and land on new platforms to build combos!"
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.MAX_STEPS = 2000
        self.WIN_Y = 20
        self.COLOR_BG = (20, 25, 40)
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # --- Game Object Lists ---
        self.player = None
        self.platforms = []
        self.ladders = []
        self.barrels = []
        self.particles = []
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.platform_combo = 0
        self.combo_multiplier = 1
        self.last_landed_platform = None
        self.prev_space_held = False
        self.barrel_spawn_timer = 0
        self.barrel_spawn_rate = 120  # Steps between spawns
        self.barrel_speed_multiplier = 1.0

        # self.reset() is called by the wrapper, no need to call it here.
    
    def _create_level(self):
        self.platforms.clear()
        self.ladders.clear()
        
        # Ground floor
        self.platforms.append(Platform(0, 360, 640, 40))
        
        # Level 1
        self.platforms.append(Platform(0, 280, 500, 20))
        self.ladders.append(Ladder(450, 280, 40, 80))
        
        # Level 2
        self.platforms.append(Platform(140, 200, 500, 20))
        self.ladders.append(Ladder(150, 200, 40, 80))
        
        # Level 3
        self.platforms.append(Platform(0, 120, 500, 20))
        self.ladders.append(Ladder(450, 120, 40, 80))
        
        # Top platform (goal)
        self.platforms.append(Platform(140, 40, 500, 20))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.platform_combo = 0
        self.combo_multiplier = 1
        self.last_landed_platform = None
        self.prev_space_held = False
        
        self.barrel_spawn_timer = 0
        self.barrel_spawn_rate = 120
        self.barrel_speed_multiplier = 1.0

        self.player = Player(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 80)
        self._create_level()
        self.barrels.clear()
        self.particles.clear()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1 # Unused
        
        space_press = space_held and not self.prev_space_held
        self.prev_space_held = space_held

        # --- Update Player ---
        landed_platform, reward_mod = self.player.update(movement, space_press, self.platforms, self.ladders)
        reward += reward_mod

        # --- Platform Combo Logic ---
        if landed_platform and landed_platform != self.last_landed_platform:
            self.platform_combo += 1
            self.score += 1 * self.combo_multiplier
            reward += 1
            self.last_landed_platform = landed_platform
            # sfx: Platform land
            
            if self.platform_combo == 3: # Speed Boost
                self.player.speed_boost_timer = 300 # 10 seconds at 30fps
                reward += 10
                # sfx: Powerup activate
            elif self.platform_combo == 5: # Invincibility
                self.player.invincible_timer = 240 # 8 seconds at 30fps
                self.platform_combo = 0 # Reset for next powerup cycle
                reward += 20
                # sfx: Invincibility activate
        elif self.player.on_ground is False and self.player.on_ladder is False:
            # If falling, reset combo unless we just jumped from a platform
             if self.last_landed_platform and not self.player.rect.colliderect(self.last_landed_platform.rect.inflate(10, 50)):
                 self.platform_combo = 0
                 self.last_landed_platform = None

        # --- Update Barrels ---
        for barrel in self.barrels:
            barrel.update(self.platforms)
        self.barrels = [b for b in self.barrels if b.y < self.SCREEN_HEIGHT + 50]

        # --- Spawn Barrels ---
        self.barrel_spawn_timer += 1
        if self.barrel_spawn_timer >= self.barrel_spawn_rate:
            self.barrel_spawn_timer = 0
            spawn_x = random.choice([150, self.SCREEN_WIDTH - 150])
            self.barrels.append(Barrel(spawn_x, -20, self.barrel_speed_multiplier))

        # --- Update Particles ---
        for p in self.particles: p.update()
        self.particles = [p for p in self.particles if p.life > 0]

        # --- Collision: Player vs Barrels ---
        terminated = False
        if self.player.invincible_timer <= 0:
            for barrel in self.barrels:
                if self.player.rect.colliderect(barrel.rect):
                    # Check for barrel jump (player falling, hits top of barrel)
                    is_jump = self.player.vy > 0 and self.player.rect.bottom < barrel.rect.centery
                    if is_jump:
                        reward += 5
                        self.score += 5 * self.combo_multiplier
                        self.combo_multiplier += 1
                        self.player.vy = self.player.jump_strength * 0.7 # Bounce
                        self.barrels.remove(barrel)
                        for _ in range(20): self.particles.append(Particle(barrel.x, barrel.y, barrel.color))
                        # sfx: Barrel break
                    else: # Fatal collision
                        reward -= 50
                        terminated = True
                        for _ in range(40): self.particles.append(Particle(self.player.rect.centerx, self.player.rect.centery, self.player.color))
                        # sfx: Player death
                        break
        
        # --- Update Difficulty ---
        if self.steps > 0 and self.steps % 200 == 0:
            self.barrel_speed_multiplier *= 1.05
            self.barrel_spawn_rate = max(30, self.barrel_spawn_rate * 0.95)

        # --- Termination Conditions ---
        if self.player.y < self.WIN_Y:
            reward += 100
            terminated = True
            # sfx: Victory
        
        truncated = False
        if self.steps >= self.MAX_STEPS:
            truncated = True

        self.game_over = terminated or truncated
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, (255, 255, 255))
        combo_text = self.font_ui.render(f"COMBO: x{self.combo_multiplier}", True, (255, 255, 255))
        
        # Draw semi-transparent background for UI text
        ui_bg = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_bg.fill((0, 0, 0, 100))
        self.screen.blit(ui_bg, (0, 0))
        
        self.screen.blit(score_text, (10, 5))
        self.screen.blit(combo_text, (self.SCREEN_WIDTH - combo_text.get_width() - 10, 5))

        # Power-up status
        status_text = ""
        if self.player.invincible_timer > 0:
            status_text = "INVINCIBLE!"
        elif self.player.speed_boost_timer > 0:
            status_text = "SPEED BOOST!"
        
        if status_text:
            status_surf = self.font_ui.render(status_text, True, (255, 255, 0))
            self.screen.blit(status_surf, (self.SCREEN_WIDTH // 2 - status_surf.get_width() // 2, 5))

    def _render_game(self):
        for ladder in self.ladders: ladder.draw(self.screen)
        for platform in self.platforms: platform.draw(self.screen, self.platform_combo, self.steps)
        for barrel in self.barrels: barrel.draw(self.screen)
        for particle in self.particles: particle.draw(self.screen)
        if not self.game_over:
            self.player.draw(self.screen, self.steps)

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
            "combo_multiplier": self.combo_multiplier,
            "platform_combo": self.platform_combo,
            "is_invincible": self.player.invincible_timer > 0,
            "has_speed_boost": self.player.speed_boost_timer > 0,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play Code ---
    # The original code had a validation check that is not needed for the final environment
    # and a main loop that requires a display. This has been adapted for clarity.
    
    # To run this in a non-headless mode, comment out the `os.environ` line at the top
    # and uncomment the `pygame.display.set_mode` line below.
    
    # For headless execution, this block will run but not display anything.
    is_headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = None
    if not is_headless:
        pygame.display.init()
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Climb and Dodge")

    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    while not terminated and not truncated:
        # --- Action Mapping for Manual Play ---
        movement = 0 # None
        space_held = 0
        shift_held = 0

        if not is_headless:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            # --- Event Handling (for quitting) ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        if not is_headless and screen:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        clock.tick(30) # Match environment's implicit FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()