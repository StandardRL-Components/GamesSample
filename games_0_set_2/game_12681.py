import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:15:20.269549
# Source Brief: brief_02681.md
# Brief Index: 2681
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Pilot a nano-sub through hostile waters, battling enemy vessels and a powerful flagship. "
        "Switch between stealth and attack modes to complete your mission."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Hold Shift to enter attack mode, "
        "and press Space to fire torpedoes."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- CRITICAL: Game Parameters ---
        self.WORLD_WIDTH = 1800
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.MAX_STEPS = 2500

        # --- EXACT SPACES ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_title = pygame.font.SysFont("monospace", 40, bold=True)

        # --- Color Palette ---
        self.COLOR_BG_DARK = (5, 10, 25)
        self.COLOR_BG_LIGHT = (10, 20, 50)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 75, 150)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_GLOW = (150, 25, 25)
        self.COLOR_FLAGSHIP = (200, 0, 100)
        self.COLOR_FLAGSHIP_GLOW = (100, 0, 50)
        self.COLOR_SONAR = (50, 255, 50)
        self.COLOR_TEXT = (220, 220, 255)
        self.COLOR_HEALTH_HIGH = (50, 200, 50)
        self.COLOR_HEALTH_LOW = (200, 50, 50)
        self.COLOR_ORANGE = (255, 165, 0)
        self.COLOR_YELLOW = (255, 255, 0)
        self.COLOR_WHITE = (255, 255, 255)

        # --- Entity State (initialized in reset) ---
        self.player_pos = None
        self.player_vel = None
        self.player_health = None
        self.player_is_large = None
        self.player_fire_cooldown = None
        
        self.enemies = None
        self.torpedoes = None
        self.particles = None
        self.sonar_pings = None
        
        self.flagship_health = None
        self.flagship_fire_cooldown = None
        
        self.camera_x = None
        self.steps = None
        self.score = None
        self.game_over_message = ""
        
        self._generate_parallax_stars()
        
        # The original code called reset() here, but it's better practice
        # to let the user call it explicitly for the first time.
        # State will be initialized on the first call to reset().
        
    def _generate_parallax_stars(self):
        self.parallax_stars = []
        for _ in range(200):
            self.parallax_stars.append({
                "pos": pygame.Vector2(random.uniform(0, self.WORLD_WIDTH), random.uniform(0, self.SCREEN_HEIGHT)),
                "depth": random.uniform(0.1, 0.8),
                "brightness": random.randint(50, 150)
            })
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.player_pos = pygame.Vector2(100, self.SCREEN_HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_health = 100
        self.player_is_large = False
        self.player_fire_cooldown = 0
        
        # Entity lists
        self.enemies = []
        self.torpedoes = []
        self.particles = []
        self.sonar_pings = []
        
        # Flagship state
        self.flagship_health = 100 # Scaled up for better gameplay feel
        self.flagship_pos = pygame.Vector2(self.WORLD_WIDTH - 100, self.SCREEN_HEIGHT / 2)
        self.flagship_fire_cooldown_max = 90 # 3 seconds
        self.flagship_fire_cooldown = self.flagship_fire_cooldown_max
        
        # Game state
        self.camera_x = 0
        self.steps = 0
        self.score = 0
        self.game_over_message = ""
        self.enemy_spawn_timer = 0
        self.enemy_speed_multiplier = 1.0

        # Initial enemies
        for _ in range(2):
            self._spawn_enemy()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action
        reward = 0.01 # Survival reward
        
        # --- 1. Handle Input ---
        self.player_is_large = (shift_held == 1)
        
        thrust = pygame.Vector2(0, 0)
        if movement == 1: thrust.y = -1 # Up
        elif movement == 2: thrust.y = 1  # Down
        elif movement == 3: thrust.x = -1 # Left
        elif movement == 4: thrust.x = 1  # Right
        
        if thrust.length() > 0:
            self.player_vel += thrust.normalize() * 0.4
            
        if space_held and self.player_is_large and self.player_fire_cooldown <= 0:
            self._fire_player_torpedo()
            self.player_fire_cooldown = 15 # 0.5 sec cooldown
            
        # --- 2. Update Game State ---
        self.steps += 1
        if self.player_fire_cooldown > 0: self.player_fire_cooldown -= 1
        
        # Update player
        self.player_pos += self.player_vel
        self.player_vel *= 0.95 # Drag
        self._clamp_player_position()
        
        # Update camera
        self.camera_x = self.player_pos.x - self.SCREEN_WIDTH / 2
        self.camera_x = max(0, min(self.camera_x, self.WORLD_WIDTH - self.SCREEN_WIDTH))

        # Update entities
        self._update_enemies()
        self._update_flagship()
        self._update_torpedoes()
        self._update_particles()
        self._update_sonar_pings()
        
        # --- 3. Handle Collisions & Rewards ---
        reward += self._handle_collisions()
        
        # --- 4. Difficulty Scaling & Spawning ---
        self._update_difficulty_and_spawns()
        
        # --- 5. Check Termination ---
        terminated = False
        truncated = False
        if self.player_health <= 0:
            terminated = True
            self.game_over_message = "MISSION FAILED"
            self._create_explosion(self.player_pos, 50, self.COLOR_PLAYER)
        elif self.flagship_health <= 0:
            terminated = True
            reward += 100 # Large victory bonus
            self.score += 1000
            self.game_over_message = "MISSION ACCOMPLISHED"
            self._create_explosion(self.flagship_pos, 100, self.COLOR_FLAGSHIP)
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            if not terminated: # Don't overwrite a victory/loss message
                self.game_over_message = "TIME LIMIT REACHED"

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # region Update Sub-methods
    def _clamp_player_position(self):
        self.player_pos.x = max(10, min(self.player_pos.x, self.WORLD_WIDTH - 10))
        self.player_pos.y = max(10, min(self.player_pos.y, self.SCREEN_HEIGHT - 10))

    def _fire_player_torpedo(self):
        # sfx: Player torpedo fire
        pos = self.player_pos.copy()
        vel = pygame.Vector2(10, 0)
        self.torpedoes.append({"pos": pos, "vel": vel, "is_player": True, "radius": 4})
        
    def _spawn_enemy(self):
        spawn_x = self.camera_x + self.SCREEN_WIDTH + 50
        if random.random() < 0.5:
             spawn_x = self.camera_x - 50
        spawn_x = max(0, min(spawn_x, self.WORLD_WIDTH))

        self.enemies.append({
            "pos": pygame.Vector2(spawn_x, random.randint(50, self.SCREEN_HEIGHT - 50)),
            "health": 30,
            "max_health": 30,
            "fire_cooldown": random.randint(60, 120),
            "patrol_dir": 1 if random.random() < 0.5 else -1,
        })
    
    def _update_difficulty_and_spawns(self):
        if self.steps % 200 == 0:
            self.enemy_speed_multiplier = min(2.5, self.enemy_speed_multiplier + 0.05)
        if self.steps > 500 and self.steps % 500 == 0:
            self.flagship_fire_cooldown_max = max(30, self.flagship_fire_cooldown_max - 5)

        self.enemy_spawn_timer -= 1
        if len(self.enemies) < 5 and self.enemy_spawn_timer <= 0:
            self._spawn_enemy()
            self.enemy_spawn_timer = 150 # 5 seconds

    def _update_enemies(self):
        for enemy in self.enemies:
            # Patrol behavior
            enemy["pos"].x += 1 * enemy["patrol_dir"] * self.enemy_speed_multiplier
            if enemy["pos"].x > self.WORLD_WIDTH - 30 or enemy["pos"].x < 30:
                enemy["patrol_dir"] *= -1
            
            # AI: Firing logic
            enemy["fire_cooldown"] -= 1
            dist_to_player = self.player_pos.distance_to(enemy["pos"])
            if self.player_is_large and dist_to_player < 300 and enemy["fire_cooldown"] <= 0:
                # sfx: Enemy torpedo fire
                direction = (self.player_pos - enemy["pos"]).normalize()
                self.torpedoes.append({
                    "pos": enemy["pos"].copy(),
                    "vel": direction * 6,
                    "is_player": False,
                    "radius": 5
                })
                enemy["fire_cooldown"] = 120 # 4 sec cooldown
            
            # AI: Sonar logic
            if self.player_is_large and random.random() < 0.01:
                self.sonar_pings.append({
                    "pos": enemy["pos"].copy(),
                    "radius": 0,
                    "max_radius": dist_to_player,
                    "life": 60
                })

    def _update_flagship(self):
        if self.flagship_health <= 0: return

        self.flagship_fire_cooldown -= 1
        dist_to_player = self.player_pos.distance_to(self.flagship_pos)
        if dist_to_player < 500 and self.flagship_fire_cooldown <= 0:
            # sfx: Flagship missile fire
            direction = (self.player_pos - self.flagship_pos).normalize()
            self.torpedoes.append({
                "pos": self.flagship_pos.copy(),
                "vel": direction * 4,
                "is_player": False,
                "radius": 8 # Larger missile
            })
            self.flagship_fire_cooldown = self.flagship_fire_cooldown_max

    def _update_torpedoes(self):
        for torpedo in self.torpedoes:
            torpedo["pos"] += torpedo["vel"]
        
        # Remove off-screen torpedoes
        self.torpedoes = [t for t in self.torpedoes if 0 < t["pos"].x < self.WORLD_WIDTH and 0 < t["pos"].y < self.SCREEN_HEIGHT]

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["vel"] *= 0.98
        self.particles = [p for p in self.particles if p["life"] > 0]
        
    def _update_sonar_pings(self):
        for ping in self.sonar_pings:
            ping["life"] -= 1
            ping["radius"] += ping["max_radius"] / 60.0
        self.sonar_pings = [p for p in self.sonar_pings if p["life"] > 0]
        
    def _handle_collisions(self):
        reward = 0
        
        # Player Torpedoes vs Enemies/Flagship
        torpedoes_to_remove = []
        for torpedo in [t for t in self.torpedoes if t["is_player"]]:
            # vs Enemies
            for enemy in self.enemies:
                if torpedo["pos"].distance_to(enemy["pos"]) < 20:
                    enemy["health"] -= 15
                    self._create_explosion(torpedo["pos"], 10, self.COLOR_ORANGE)
                    torpedoes_to_remove.append(torpedo)
                    break
            else: # vs Flagship
                if self.flagship_health > 0 and torpedo["pos"].distance_to(self.flagship_pos) < 50:
                    self.flagship_health -= 5
                    self._create_explosion(torpedo["pos"], 20, self.COLOR_ORANGE)
                    torpedoes_to_remove.append(torpedo)
        
        self.torpedoes = [t for t in self.torpedoes if t not in torpedoes_to_remove]

        # Enemy Torpedoes vs Player
        torpedoes_to_remove = []
        for torpedo in [t for t in self.torpedoes if not t["is_player"]]:
            player_radius = 20 if self.player_is_large else 10
            if torpedo["pos"].distance_to(self.player_pos) < player_radius + torpedo["radius"]:
                self.player_health -= 10
                reward -= 5
                self._create_explosion(torpedo["pos"], 15, self.COLOR_PLAYER)
                torpedoes_to_remove.append(torpedo)
                break
        
        self.torpedoes = [t for t in self.torpedoes if t not in torpedoes_to_remove]
                
        # Cleanup dead enemies
        dead_enemies = [e for e in self.enemies if e["health"] <= 0]
        for dead in dead_enemies:
            self._create_explosion(dead["pos"], 30, self.COLOR_ENEMY)
            reward += 10
            self.score += 100
            self.enemies.remove(dead)
            
        return reward

    def _create_explosion(self, pos, num_particles, base_color):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": random.randint(20, 40),
                "radius": random.uniform(1, 4),
                "color": random.choice([base_color, self.COLOR_ORANGE, self.COLOR_YELLOW, self.COLOR_WHITE])
            })
    # endregion

    # region Rendering Methods
    def _get_observation(self):
        self._render_background()
        self._render_game_objects()
        self._render_effects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Gradient background
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_DARK[0] * (1 - interp) + self.COLOR_BG_LIGHT[0] * interp,
                self.COLOR_BG_DARK[1] * (1 - interp) + self.COLOR_BG_LIGHT[1] * interp,
                self.COLOR_BG_DARK[2] * (1 - interp) + self.COLOR_BG_LIGHT[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

        # Parallax stars
        for star in self.parallax_stars:
            x = (star["pos"].x - self.camera_x * star["depth"]) % self.WORLD_WIDTH
            y = star["pos"].y
            if 0 <= x < self.SCREEN_WIDTH:
                c = star["brightness"]
                self.screen.set_at((int(x), int(y)), (c, c, c+20))

    def _render_game_objects(self):
        # Render Flagship
        if self.flagship_health > 0:
            self._draw_flagship(self.flagship_pos - pygame.Vector2(self.camera_x, 0))

        # Render Enemies
        for enemy in self.enemies:
            self._draw_enemy(enemy, enemy["pos"] - pygame.Vector2(self.camera_x, 0))
            
        # Render Player
        if self.player_health > 0:
            self._draw_player(self.player_pos - pygame.Vector2(self.camera_x, 0))

        # Render Torpedoes
        for torpedo in self.torpedoes:
            color = self.COLOR_PLAYER if torpedo["is_player"] else self.COLOR_ENEMY
            pos = torpedo["pos"] - pygame.Vector2(self.camera_x, 0)
            self._draw_glow_circle(pos, torpedo["radius"], color, (color[0]//2, color[1]//2, color[2]//2))

    def _render_effects(self):
        # Render Sonar Pings
        for ping in self.sonar_pings:
            pos = ping["pos"] - pygame.Vector2(self.camera_x, 0)
            alpha = max(0, 255 * (ping["life"] / 60.0))
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(ping["radius"]), (*self.COLOR_SONAR, int(alpha)))
            
        # Render Particles
        for p in self.particles:
            pos = p["pos"] - pygame.Vector2(self.camera_x, 0)
            alpha = max(0, 255 * (p["life"] / 40.0))
            color = (*p["color"], int(alpha))
            pygame.draw.circle(self.screen, color, (int(pos.x), int(pos.y)), int(p["radius"]))
            
    def _render_ui(self):
        # Player Health Bar
        self._draw_health_bar(pygame.Rect(10, 10, 200, 20), self.player_health / 100, "NANO-SUB INTEGRITY")
        
        # Flagship Health Bar
        if self.flagship_health > 0:
            self._draw_health_bar(pygame.Rect(self.SCREEN_WIDTH - 210, 10, 200, 20), self.flagship_health / 100, "FLAGSHIP HULL")
            
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 10))

        # Game Over Message
        if self.game_over_message:
            title_text = self.font_title.render(self.game_over_message, True, self.COLOR_WHITE)
            text_rect = title_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            pygame.draw.rect(self.screen, (0,0,0,150), text_rect.inflate(20,20))
            self.screen.blit(title_text, text_rect)
    
    def _draw_player(self, pos):
        if self.player_health <= 0: return
        
        size_indicator_pos = pos + pygame.Vector2(0, -30)
        size_indicator_width = 30
        
        if self.player_is_large:
            radius = 20
            glow_radius = 40
            # Draw size indicator bar
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, (int(size_indicator_pos.x - size_indicator_width/2), int(size_indicator_pos.y), size_indicator_width, 4))
        else:
            radius = 10
            glow_radius = 20
            # Draw empty size indicator bar
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_GLOW, (int(size_indicator_pos.x - size_indicator_width/2), int(size_indicator_pos.y), size_indicator_width, 4), 1)

        self._draw_glow_circle(pos, radius, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW, glow_radius)

    def _draw_enemy(self, enemy, pos):
        radius = 15
        self._draw_glow_circle(pos, radius, self.COLOR_ENEMY, self.COLOR_ENEMY_GLOW, radius * 2)
        # Health bar above enemy
        health_pct = enemy["health"] / enemy["max_health"]
        bar_width = 30
        bar_height = 4
        bar_pos = pos - pygame.Vector2(bar_width / 2, radius + 10)
        pygame.draw.rect(self.screen, self.COLOR_ENEMY_GLOW, (int(bar_pos.x), int(bar_pos.y), bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_HIGH, (int(bar_pos.x), int(bar_pos.y), int(bar_width * health_pct), bar_height))

    def _draw_flagship(self, pos):
        width, height = 100, 80
        rect = pygame.Rect(pos.x - width/2, pos.y - height/2, width, height)
        
        # Glow
        glow_rect = rect.inflate(20, 20)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, (*self.COLOR_FLAGSHIP_GLOW, 50), s.get_rect(), border_radius=15)
        self.screen.blit(s, glow_rect.topleft)
        
        # Body
        pygame.draw.rect(self.screen, self.COLOR_FLAGSHIP, rect, border_radius=10)
        pygame.draw.rect(self.screen, self.COLOR_WHITE, rect, width=2, border_radius=10)

    def _draw_glow_circle(self, pos, radius, color, glow_color, glow_radius=None):
        if glow_radius is None:
            glow_radius = radius * 2
        
        # Glow
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*glow_color, 80), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (int(pos.x - glow_radius), int(pos.y - glow_radius)))
        
        # Core circle
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(radius), color)
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius), color)

    def _draw_health_bar(self, rect, percentage, label):
        percentage = max(0, min(1, percentage))
        
        # Background
        pygame.draw.rect(self.screen, (50,50,50), rect, border_radius=4)
        
        # Fill
        fill_color = (
            self.COLOR_HEALTH_LOW[0] + (self.COLOR_HEALTH_HIGH[0] - self.COLOR_HEALTH_LOW[0]) * percentage,
            self.COLOR_HEALTH_LOW[1] + (self.COLOR_HEALTH_HIGH[1] - self.COLOR_HEALTH_LOW[1]) * percentage,
            self.COLOR_HEALTH_LOW[2] + (self.COLOR_HEALTH_HIGH[2] - self.COLOR_HEALTH_LOW[2]) * percentage
        )
        fill_rect = pygame.Rect(rect.left, rect.top, int(rect.width * percentage), rect.height)
        pygame.draw.rect(self.screen, fill_color, fill_rect, border_radius=4)
        
        # Border
        pygame.draw.rect(self.screen, self.COLOR_TEXT, rect, 2, border_radius=4)
        
        # Label
        text = self.font_ui.render(label, True, self.COLOR_TEXT)
        self.screen.blit(text, (rect.x + 5, rect.y + 2))
    # endregion

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "flagship_health": self.flagship_health,
            "enemies_remaining": len(self.enemies)
        }
        
    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # Set a non-dummy driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    
    # Manual play loop
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Nano Sub")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        # --- Action mapping for human input ---
        movement = 0 # None
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    
    # Display final screen for a few seconds
    pygame.time.wait(3000)
    
    env.close()