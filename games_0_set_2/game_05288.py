
# Generated: 2025-08-28T04:33:39.233997
# Source Brief: brief_05288.md
# Brief Index: 5288

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Helper classes for game entities
class Player:
    """Represents the player character."""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 30
        self.height = 40
        self.vx = 0
        self.vy = 0
        self.health = 100
        self.max_health = 100
        self.on_ground = False
        self.shoot_cooldown = 0
        self.invincibility_timer = 0
        self.facing_right = True
        self.is_shooting = False

    @property
    def rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def take_damage(self, amount):
        if self.invincibility_timer <= 0:
            self.health -= amount
            self.invincibility_timer = 30 # 1 second invincibility at 30fps
            # Sound effect placeholder: player_hit.wav

    def draw(self, screen, camera_x):
        screen_x = int(self.x - camera_x)
        screen_y = int(self.y)
        
        # Invincibility flash
        if self.invincibility_timer > 0 and self.invincibility_timer % 4 < 2:
            return # Don't draw to create a flashing effect

        # Body
        body_color = (0, 220, 0)
        pygame.draw.rect(screen, body_color, (screen_x, screen_y, self.width, self.height), border_radius=3)
        
        # Eye
        eye_x = screen_x + (self.width * 0.7 if self.facing_right else self.width * 0.3)
        eye_y = screen_y + self.height * 0.3
        pygame.draw.circle(screen, (255, 255, 255), (int(eye_x), int(eye_y)), 5)
        pygame.draw.circle(screen, (0, 0, 0), (int(eye_x), int(eye_y)), 2)

        # Muzzle flash
        if self.is_shooting:
            flash_x = screen_x + (self.width if self.facing_right else 0)
            flash_y = screen_y + self.height * 0.6
            points = []
            for i in range(8):
                angle = i * math.pi / 4
                outer_r, inner_r = 20, 10
                r = outer_r if i % 2 == 0 else inner_r
                points.append((flash_x + r * math.cos(angle), flash_y + r * math.sin(angle)))
            pygame.draw.polygon(screen, (255, 255, 0), points)
            pygame.draw.polygon(screen, (255, 128, 0), points, 2)
            self.is_shooting = False

class Enemy:
    """Base class for all enemy types."""
    def __init__(self, x, y, enemy_type, health, speed_multiplier):
        self.x = x
        self.y = y
        self.type = enemy_type
        self.health = health
        self.max_health = health
        self.speed_multiplier = speed_multiplier
        self.initial_x = x
        self.initial_y = y
        self.anim_timer = random.uniform(0, 2 * math.pi)

        if self.type == "ground":
            self.width, self.height = 35, 35
            self.color = (200, 50, 50)
            self.vx = 1.5 * self.speed_multiplier
        elif self.type == "flying":
            self.width, self.height = 40, 25
            self.color = (220, 100, 50)
            self.vx = 2.0 * self.speed_multiplier
        elif self.type == "turret":
            self.width, self.height = 30, 30
            self.color = (180, 50, 180)
            self.shoot_cooldown = random.randint(60, 120)

    @property
    def rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def update(self):
        self.anim_timer += 0.1
        if self.type == "ground":
            self.x += self.vx
            if abs(self.x - self.initial_x) > 100:
                self.vx *= -1
        elif self.type == "flying":
            self.x += self.vx
            self.y = self.initial_y + math.sin(self.anim_timer) * 40
            if abs(self.x - self.initial_x) > 150:
                self.vx *= -1
        elif self.type == "turret":
            self.shoot_cooldown -= 1
            if self.shoot_cooldown <= 0:
                self.shoot_cooldown = 120 # 4 seconds at 30fps
                return True # Indicates it wants to shoot
        return False

    def draw(self, screen, camera_x):
        screen_x = int(self.x - camera_x)
        screen_y = int(self.y)
        
        if self.type == "ground":
            pygame.draw.rect(screen, self.color, (screen_x, screen_y, self.width, self.height))
            # Spikes
            for i in range(4):
                pygame.draw.polygon(screen, (150, 30, 30), [
                    (screen_x + i * 9 + 2, screen_y),
                    (screen_x + i * 9 + 4, screen_y - 5),
                    (screen_x + i * 9 + 6, screen_y)
                ])
        elif self.type == "flying":
            # Body
            pygame.draw.ellipse(screen, self.color, (screen_x, screen_y, self.width, self.height))
            # Propeller
            prop_y = screen_y + self.height / 2
            pygame.draw.line(screen, (150, 150, 150), (screen_x - 5, prop_y), (screen_x + self.width + 5, prop_y), 3)
        elif self.type == "turret":
            base_rect = pygame.Rect(screen_x, screen_y, self.width, self.height)
            pygame.draw.rect(screen, self.color, base_rect)
            pygame.draw.rect(screen, (100, 30, 100), base_rect, 3)
            # Barrel
            pygame.draw.rect(screen, self.color, (screen_x + 10, screen_y + self.height, 10, 10))

class Projectile:
    """Represents projectiles fired by player or enemies."""
    def __init__(self, x, y, vx, color, owner):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = 0
        self.width = 15
        self.height = 5
        self.color = color
        self.owner = owner

    @property
    def rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def update(self):
        self.x += self.vx
        self.y += self.vy

    def draw(self, screen, camera_x):
        screen_x = int(self.x - camera_x)
        screen_y = int(self.y)
        pygame.draw.rect(screen, self.color, (screen_x, screen_y, self.width, self.height), border_radius=2)
        # Glow effect
        glow_rect = pygame.Rect(screen_x - 2, screen_y - 2, self.width + 4, self.height + 4)
        pygame.draw.rect(screen, self.color, glow_rect, width=1, border_radius=3)

class Particle:
    """Represents a single particle for effects like explosions."""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-5, 1)
        self.lifespan = random.randint(15, 30)
        self.color = random.choice([(255, 50, 0), (255, 150, 0), (255, 255, 0)])
        self.size = random.randint(4, 8)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.3 # Gravity
        self.lifespan -= 1
        self.size = max(0, self.size - 0.2)

    def draw(self, screen, camera_x):
        if self.lifespan > 0:
            screen_x = int(self.x - camera_x)
            screen_y = int(self.y)
            pygame.draw.rect(screen, self.color, (screen_x, screen_y, int(self.size), int(self.size)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑ to jump, ←→ to move. Hold space to fire your weapon."
    )

    game_description = (
        "Control a jumping, shooting robot to blast through waves of enemies and reach the end of each stage in this side-scrolling action game."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.STAGE_WIDTH = 2560
        self.TOTAL_WIDTH = self.STAGE_WIDTH * 3
        self.FPS = 30
        self.MAX_STEPS = 5400 # 30fps * 60s * 3 stages

        # Colors
        self.COLOR_BG = (20, 30, 50)
        self.COLOR_BG_LAYER2 = (30, 45, 70)
        self.COLOR_GROUND = (60, 60, 80)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_HEALTH = (0, 255, 0)
        self.COLOR_HEALTH_BG = (255, 0, 0)
        self.COLOR_PORTAL = (255, 220, 0)

        # Physics
        self.GRAVITY = 0.8
        self.PLAYER_SPEED = 5
        self.JUMP_STRENGTH = -15
        self.PROJECTILE_SPEED = 12
        self.PLAYER_SHOOT_COOLDOWN = 10 # 3 shots per second

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Game State (initialized in reset)
        self.player = None
        self.platforms = []
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.stage_portals = []
        
        self.steps = 0
        self.score = 0
        self.stage = 1
        self.time_left = 0
        self.camera_x = 0
        self.game_over = False
        self.victory = False
        
        self.reset()
        self.validate_implementation()
    
    def _generate_world(self):
        self.platforms = []
        self.enemies = []
        self.stage_portals = []
        
        # Ground floor for all stages
        self.platforms.append(pygame.Rect(0, self.HEIGHT - 20, self.TOTAL_WIDTH, 20))
        
        speed_multiplier = 1.0 + (self.stage - 1) * 0.05
        
        # Stage 1
        for i in range(5):
            self.platforms.append(pygame.Rect(400 + i * 400, self.HEIGHT - 100, 150, 20))
        self._spawn_enemies_for_stage(1, speed_multiplier)
        self.stage_portals.append(pygame.Rect(self.STAGE_WIDTH - 100, self.HEIGHT - 120, 20, 100))
        
        # Stage 2
        for i in range(4):
            self.platforms.append(pygame.Rect(self.STAGE_WIDTH + 500 + i * 450, self.HEIGHT - 80 - (i%2 * 60), 120, 20))
        self._spawn_enemies_for_stage(2, speed_multiplier)
        self.stage_portals.append(pygame.Rect(self.STAGE_WIDTH * 2 - 100, self.HEIGHT - 120, 20, 100))

        # Stage 3
        self.platforms.append(pygame.Rect(self.STAGE_WIDTH * 2 + 300, self.HEIGHT - 120, 800, 20))
        self.platforms.append(pygame.Rect(self.STAGE_WIDTH * 2 + 600, self.HEIGHT - 220, 200, 20))
        self._spawn_enemies_for_stage(3, speed_multiplier)
        self.stage_portals.append(pygame.Rect(self.TOTAL_WIDTH - 100, self.HEIGHT - 120, 20, 100))

    def _spawn_enemies_for_stage(self, stage_num, speed_multiplier):
        stage_start_x = (stage_num - 1) * self.STAGE_WIDTH
        
        enemy_configs = {
            1: [("ground", 600, self.HEIGHT - 55), ("ground", 1200, self.HEIGHT - 55), ("flying", 1800, 150)],
            2: [("ground", stage_start_x + 400, self.HEIGHT - 55), ("flying", stage_start_x + 800, 120), ("turret", stage_start_x + 1500, self.HEIGHT - 50), ("flying", stage_start_x + 2000, 180)],
            3: [("turret", stage_start_x + 350, self.HEIGHT - 140), ("turret", stage_start_x + 1050, self.HEIGHT - 140), ("ground", stage_start_x + 700, self.HEIGHT - 155), ("flying", stage_start_x + 1400, 100), ("flying", stage_start_x + 1800, 200), ("ground", stage_start_x + 2200, self.HEIGHT - 55)],
        }
        
        for config in enemy_configs.get(stage_num, []):
            e_type, x, y = config
            health = {"ground": 20, "flying": 30, "turret": 50}[e_type]
            self.enemies.append(Enemy(x, y, e_type, health, speed_multiplier))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player = Player(100, self.HEIGHT - 60)
        
        self.steps = 0
        self.score = 0
        self.stage = 1
        self.time_left = 60 * 3 # 60s per stage
        self.game_over = False
        self.victory = False
        
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        
        self._generate_world()
        
        self.camera_x = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Action Handling & Action Rewards ---
        if self.game_over or self.victory:
            movement = 0
            space_held = False
        
        player_action_taken = False
        if movement == 3: # Left
            self.player.vx = -self.PLAYER_SPEED
            self.player.facing_right = False
            player_action_taken = True
            reward -= 0.02
        elif movement == 4: # Right
            self.player.vx = self.PLAYER_SPEED
            self.player.facing_right = True
            reward += 0.1
            player_action_taken = True
        else:
            self.player.vx = 0

        if movement == 1 and self.player.on_ground: # Jump
            self.player.vy = self.JUMP_STRENGTH
            self.player.on_ground = False
            player_action_taken = True
            reward -= 0.02
            # Sound effect placeholder: jump.wav
        
        if space_held and self.player.shoot_cooldown <= 0:
            self.player.shoot_cooldown = self.PLAYER_SHOOT_COOLDOWN
            px = self.player.x + self.player.width if self.player.facing_right else self.player.x - 15
            py = self.player.y + self.player.height * 0.55
            p_vx = self.PROJECTILE_SPEED if self.player.facing_right else -self.PROJECTILE_SPEED
            self.player_projectiles.append(Projectile(px, py, p_vx, (100, 150, 255), "player"))
            self.player.is_shooting = True
            player_action_taken = True
            reward -= 0.02
            # Sound effect placeholder: laser_shoot.wav

        # --- Game Logic Update ---
        self.steps += 1
        self.time_left -= 1.0 / self.FPS
        
        # Update player
        self.player.x += self.player.vx
        self.player.x = max(0, min(self.player.x, self.TOTAL_WIDTH - self.player.width))
        self.player.vy += self.GRAVITY
        self.player.y += self.player.vy
        self.player.on_ground = False
        self.player.shoot_cooldown = max(0, self.player.shoot_cooldown - 1)
        self.player.invincibility_timer = max(0, self.player.invincibility_timer - 1)

        # Player-platform collision
        player_rect = self.player.rect
        for plat in self.platforms:
            if player_rect.colliderect(plat) and self.player.vy > 0:
                if (player_rect.bottom - self.player.vy) <= plat.top:
                    self.player.y = plat.top - self.player.height
                    self.player.vy = 0
                    self.player.on_ground = True
                    break

        # Update entities
        for p in self.player_projectiles: p.update()
        for p in self.enemy_projectiles: p.update()
        for p in self.particles: p.update()

        for e in self.enemies:
            if e.update(): # Turret wants to shoot
                ex, ey = e.x + e.width / 2, e.y + e.height
                proj = Projectile(ex, ey, 0, (255, 100, 100), "enemy")
                proj.vy = 6
                self.enemy_projectiles.append(proj)
                # Sound effect placeholder: enemy_shoot.wav
        
        # --- Collisions ---
        for p in self.player_projectiles[:]:
            for e in self.enemies[:]:
                if p.rect.colliderect(e.rect):
                    e.health -= 10
                    reward += 1.0 # Hit reward
                    if p in self.player_projectiles: self.player_projectiles.remove(p)
                    if e.health <= 0:
                        reward += 10.0 # Kill reward
                        self.score += 100
                        for _ in range(20): self.particles.append(Particle(e.x + e.width/2, e.y + e.height/2))
                        self.enemies.remove(e)
                        # Sound effect placeholder: explosion.wav
                    break

        for p in self.enemy_projectiles[:]:
            if p.rect.colliderect(self.player.rect):
                self.player.take_damage(15)
                self.enemy_projectiles.remove(p)

        for e in self.enemies:
            if e.rect.colliderect(self.player.rect):
                self.player.take_damage(10)
        
        # --- Stage Progression ---
        current_stage_portal = self.stage_portals[self.stage - 1]
        if self.player.rect.colliderect(current_stage_portal):
            if self.stage < 3:
                self.stage += 1
                self.player.x = (self.stage - 1) * self.STAGE_WIDTH + 100
                self.player.y = self.HEIGHT - 60
                self.player.vx = self.player.vy = 0
                self.score += 500
                reward += 50.0
                # Sound effect placeholder: stage_clear.wav
            else:
                self.victory = True
                self.score += 1000
                reward += 100.0
                # Sound effect placeholder: victory.wav

        # --- Cleanup ---
        self.player_projectiles = [p for p in self.player_projectiles if 0 < p.x - self.camera_x < self.WIDTH]
        self.enemy_projectiles = [p for p in self.enemy_projectiles if 0 < p.y < self.HEIGHT]
        self.particles = [p for p in self.particles if p.lifespan > 0]
        
        # --- Termination Check ---
        if self.player.health <= 0: self.game_over = True
        if self.time_left <= 0: self.game_over = True
        if self.steps >= self.MAX_STEPS: self.game_over = True

        terminated = self.game_over or self.victory
        
        # Update camera
        target_camera_x = self.player.x - self.WIDTH / 3
        stage_start_x = (self.stage - 1) * self.STAGE_WIDTH
        stage_end_x = self.stage * self.STAGE_WIDTH - self.WIDTH
        self.camera_x = max(stage_start_x, min(target_camera_x, stage_end_x))

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _render_background(self):
        # Parallax background
        for i in range(5):
            x = (i * 300 - self.camera_x * 0.5) % (self.WIDTH + 300) - 300
            pygame.draw.rect(self.screen, self.COLOR_BG_LAYER2, (x, 100, 10, 300))
            pygame.draw.rect(self.screen, self.COLOR_BG_LAYER2, (x + 150, 200, 150, 10))

    def _render_game(self):
        # Render platforms
        for plat in self.platforms:
            if plat.right > self.camera_x and plat.left < self.camera_x + self.WIDTH:
                screen_rect = plat.move(-self.camera_x, 0)
                pygame.draw.rect(self.screen, self.COLOR_GROUND, screen_rect)

        # Render portals
        for i, portal in enumerate(self.stage_portals):
            if portal.right > self.camera_x and portal.left < self.camera_x + self.WIDTH:
                screen_rect = portal.move(-self.camera_x, 0)
                pulse = abs(math.sin(self.steps * 0.1))
                for j in range(5, 0, -1):
                    alpha = 150 - j * 25
                    color = (*self.COLOR_PORTAL, alpha)
                    radius = 10 + j * 4 + pulse * 5
                    temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                    pygame.gfxdraw.filled_circle(temp_surf, int(radius), int(radius), int(radius), color)
                    self.screen.blit(temp_surf, (screen_rect.centerx - radius, screen_rect.centery - radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Render entities
        for e in self.enemies: e.draw(self.screen, self.camera_x)
        for p in self.player_projectiles: p.draw(self.screen, self.camera_x)
        for p in self.enemy_projectiles: p.draw(self.screen, self.camera_x)
        self.player.draw(self.screen, self.camera_x)
        for p in self.particles: p.draw(self.screen, self.camera_x)

    def _render_ui(self):
        health_ratio = max(0, self.player.health / self.player.max_health)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, (10, 10, 200 * health_ratio, 20))
        
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        time_display = max(0, int(self.time_left))
        stage_text = self.font_small.render(f"STAGE: {self.stage} | TIME: {time_display}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.WIDTH/2 - stage_text.get_width()/2, self.HEIGHT - 30))

        if self.game_over:
            text = self.font_large.render("GAME OVER", True, self.COLOR_HEALTH_BG)
            self.screen.blit(text, (self.WIDTH/2 - text.get_width()/2, self.HEIGHT/2 - text.get_height()/2))
        elif self.victory:
            text = self.font_large.render("VICTORY!", True, self.COLOR_PORTAL)
            self.screen.blit(text, (self.WIDTH/2 - text.get_width()/2, self.HEIGHT/2 - text.get_height()/2))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "health": self.player.health,
            "time_left": self.time_left,
            "victory": self.victory,
        }
    
    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    import os
    if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Robo-Blaster")
    
    obs, info = env.reset()
    terminated = False
    
    while not terminated:
        movement, space_held, shift_held = 0, 0, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_UP]: movement = 1
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(env.FPS)
        
    env.close()
    print(f"Game Over! Final Score: {info['score']}")