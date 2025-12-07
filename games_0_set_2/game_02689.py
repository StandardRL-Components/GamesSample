
# Generated: 2025-08-28T05:39:50.766555
# Source Brief: brief_02689.md
# Brief Index: 2689

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold space to shoot. Survive for 60 seconds."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a relentless zombie horde for 60 seconds in this top-down isometric shooter. Kill zombies for score and don't let them touch you!"
    )

    # Frames auto-advance at a fixed rate for time-based gameplay.
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    GAME_DURATION_SECONDS = 60

    # Colors
    COLOR_BG = (25, 25, 30)
    COLOR_GROUND_TILE = (40, 40, 45)
    COLOR_PLAYER = (50, 200, 50)
    COLOR_ZOMBIE = (120, 100, 80)
    COLOR_ZOMBIE_DARK = (90, 70, 50)
    COLOR_PROJECTILE = (255, 255, 100)
    COLOR_SHADOW = (15, 15, 20)
    COLOR_BLOOD = (180, 0, 0)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_HEALTH_BAR_BG = (80, 20, 20)
    COLOR_HEALTH_BAR_FILL = (220, 30, 30)
    
    # Player settings
    PLAYER_SPEED = 2.5
    PLAYER_RADIUS = 10
    PLAYER_HEALTH_MAX = 100
    PLAYER_FIRE_RATE = 8  # frames between shots
    PLAYER_HIT_INVULNERABILITY = 30 # frames

    # Zombie settings
    ZOMBIE_SPEED = 0.75
    ZOMBIE_RADIUS = 11
    ZOMBIE_HEALTH = 40
    ZOMBIE_DAMAGE = 10
    ZOMBIE_MAX_COUNT = 50
    ZOMBIE_INITIAL_SPAWN_RATE = 25 # frames
    ZOMBIE_SPAWN_RATE_ACCELERATION = 120 # spawn rate decreases every X frames
    ZOMBIE_MIN_SPAWN_RATE = 8

    # Projectile settings
    PROJECTILE_SPEED = 8
    PROJECTILE_RADIUS = 3
    PROJECTILE_DAMAGE = 20

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Initialize state variables to be defined in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_timer = 0
        self.player_pos = None
        self.player_health = 0
        self.player_facing_direction = None
        self.player_hit_cooldown = 0
        self.fire_cooldown = 0
        self.zombies = []
        self.projectiles = []
        self.particles = []
        self.zombie_spawn_timer = 0
        self.zombie_spawn_rate = 0
        self.muzzle_flash_timer = 0
        
        self.np_random = None

        self.validate_implementation()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_timer = self.GAME_DURATION_SECONDS
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_health = self.PLAYER_HEALTH_MAX
        self.player_facing_direction = pygame.Vector2(0, -1) # Start facing up
        self.player_hit_cooldown = 0
        self.fire_cooldown = 0
        
        self.zombies = []
        self.projectiles = []
        self.particles = []
        
        self.zombie_spawn_timer = self.ZOMBIE_INITIAL_SPAWN_RATE
        self.zombie_spawn_rate = self.ZOMBIE_INITIAL_SPAWN_RATE
        
        self.muzzle_flash_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action
        reward = 0.0

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        # --- Update Timers ---
        self.steps += 1
        self.game_timer = max(0, self.game_timer - 1 / self.FPS)
        self.fire_cooldown = max(0, self.fire_cooldown - 1)
        self.player_hit_cooldown = max(0, self.player_hit_cooldown - 1)
        self.muzzle_flash_timer = max(0, self.muzzle_flash_timer - 1)

        # --- Handle Player Actions ---
        self._handle_input(movement, space_held)
        
        # --- Update Game Logic ---
        self._update_projectiles()
        zombies_killed = self._update_zombies()
        self._update_particles()
        self._spawn_zombies()
        
        # --- Calculate Rewards ---
        reward += 0.01  # Small survival reward per frame
        reward += zombies_killed * 1.0 # Reward for killing a zombie

        # --- Check Termination ---
        terminated = self.player_health <= 0 or self.game_timer <= 0
        if terminated:
            self.game_over = True
            if self.player_health > 0: # Survived
                reward += 100.0
            else: # Died
                reward -= 100.0

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        player_velocity = pygame.Vector2(0, 0)
        if movement == 1: player_velocity.y = -1
        elif movement == 2: player_velocity.y = 1
        elif movement == 3: player_velocity.x = -1
        elif movement == 4: player_velocity.x = 1

        if player_velocity.length() > 0:
            player_velocity.normalize_ip()
            self.player_facing_direction = player_velocity.copy()
            self.player_pos += player_velocity * self.PLAYER_SPEED

        # Clamp player position to screen bounds
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)
        
        # Shooting
        if space_held and self.fire_cooldown == 0:
            self._fire_projectile()

    def _fire_projectile(self):
        # sfx: Laser_Shoot
        self.fire_cooldown = self.PLAYER_FIRE_RATE
        self.muzzle_flash_timer = 2 # frames
        
        start_pos = self.player_pos + self.player_facing_direction * (self.PLAYER_RADIUS + 5)
        projectile = {
            "pos": start_pos,
            "vel": self.player_facing_direction * self.PROJECTILE_SPEED,
        }
        self.projectiles.append(projectile)

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p["pos"] += p["vel"]
            if not (0 < p["pos"].x < self.WIDTH and 0 < p["pos"].y < self.HEIGHT):
                self.projectiles.remove(p)

    def _update_zombies(self):
        killed_count = 0
        for z in self.zombies[:]:
            # Movement
            direction = (self.player_pos - z["pos"]).normalize()
            z["pos"] += direction * self.ZOMBIE_SPEED

            # Collision with player
            if z["pos"].distance_to(self.player_pos) < self.ZOMBIE_RADIUS + self.PLAYER_RADIUS:
                if self.player_hit_cooldown == 0:
                    self.player_health = max(0, self.player_health - self.ZOMBIE_DAMAGE)
                    self.player_hit_cooldown = self.PLAYER_HIT_INVULNERABILITY
                    # sfx: Player_Hurt

            # Collision with projectiles
            for p in self.projectiles[:]:
                if z["pos"].distance_to(p["pos"]) < self.ZOMBIE_RADIUS + self.PROJECTILE_RADIUS:
                    self.projectiles.remove(p)
                    z["health"] -= self.PROJECTILE_DAMAGE
                    self._create_blood_splatter(z["pos"], 5, 0.5)
                    # sfx: Zombie_Hit
                    if z["health"] <= 0:
                        self.zombies.remove(z)
                        self.score += 10
                        killed_count += 1
                        self._create_blood_splatter(z["pos"], 20, 1.0)
                        # sfx: Zombie_Die
                        break # Zombie is dead, no need to check more projectiles
        return killed_count

    def _spawn_zombies(self):
        self.zombie_spawn_timer -= 1
        if self.zombie_spawn_timer <= 0 and len(self.zombies) < self.ZOMBIE_MAX_COUNT:
            # Difficulty scaling
            self.zombie_spawn_rate = max(self.ZOMBIE_MIN_SPAWN_RATE, self.ZOMBIE_INITIAL_SPAWN_RATE - (self.steps // self.ZOMBIE_SPAWN_RATE_ACCELERATION))
            self.zombie_spawn_timer = self.zombie_spawn_rate
            
            # Spawn off-screen
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), -self.ZOMBIE_RADIUS)
            elif edge == 1: # Bottom
                pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ZOMBIE_RADIUS)
            elif edge == 2: # Left
                pos = pygame.Vector2(-self.ZOMBIE_RADIUS, self.np_random.uniform(0, self.HEIGHT))
            else: # Right
                pos = pygame.Vector2(self.WIDTH + self.ZOMBIE_RADIUS, self.np_random.uniform(0, self.HEIGHT))
            
            self.zombies.append({"pos": pos, "health": self.ZOMBIE_HEALTH})

    def _create_blood_splatter(self, pos, num_particles, speed_multiplier):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_multiplier
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifetime = self.np_random.integers(30, 60)
            self.particles.append({"pos": pos.copy(), "vel": vel, "lifetime": lifetime, "max_lifetime": lifetime})

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.9 # friction
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "timer": self.game_timer}

    def _render_game(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        for i in range(0, self.WIDTH, 40):
            for j in range(0, self.HEIGHT, 40):
                if (i // 40 + j // 40) % 2 == 0:
                    pygame.draw.rect(self.screen, self.COLOR_GROUND_TILE, (i, j, 40, 40))

        # --- Particles ---
        for p in self.particles:
            alpha = int(200 * (p["lifetime"] / p["max_lifetime"]))
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), 2, (*self.COLOR_BLOOD, alpha))

        # --- Entities (sorted by Y for pseudo-3D) ---
        entities = [{"type": "zombie", "pos": z["pos"], "radius": self.ZOMBIE_RADIUS} for z in self.zombies]
        entities.append({"type": "player", "pos": self.player_pos, "radius": self.PLAYER_RADIUS})
        entities.sort(key=lambda e: e["pos"].y)

        for entity in entities:
            pos = entity["pos"]
            radius = entity["radius"]
            shadow_offset = radius * 0.5
            
            # Shadow
            pygame.gfxdraw.filled_ellipse(self.screen, int(pos.x), int(pos.y + shadow_offset), int(radius), int(radius * 0.5), self.COLOR_SHADOW)
            
            # Body
            if entity["type"] == "player":
                color = self.COLOR_PLAYER
                if self.player_hit_cooldown > 0 and (self.player_hit_cooldown // 4) % 2 == 0:
                    color = (255, 255, 255) # Flash white when hit
                pygame.gfxdraw.filled_ellipse(self.screen, int(pos.x), int(pos.y), radius, radius, color)
                pygame.gfxdraw.aaellipse(self.screen, int(pos.x), int(pos.y), radius, radius, color)
            else: # Zombie
                pygame.gfxdraw.filled_ellipse(self.screen, int(pos.x), int(pos.y), radius, radius, self.COLOR_ZOMBIE)
                pygame.gfxdraw.aaellipse(self.screen, int(pos.x), int(pos.y), radius, radius, self.COLOR_ZOMBIE_DARK)

        # --- Projectiles ---
        for p in self.projectiles:
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, p["pos"], p["pos"] + p["vel"] * 0.5, 3)
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)

        # --- Muzzle Flash ---
        if self.muzzle_flash_timer > 0:
            flash_pos = self.player_pos + self.player_facing_direction * (self.PLAYER_RADIUS + 3)
            pygame.gfxdraw.filled_circle(self.screen, int(flash_pos.x), int(flash_pos.y), 6, (255, 255, 200))

        # --- UI ---
        self._render_ui()
        
        # --- Game Over Screen ---
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            result_text = "YOU SURVIVED!" if self.player_health > 0 else "YOU DIED"
            text_surface = self.font_large.render(result_text, True, self.COLOR_UI_TEXT)
            text_rect = text_surface.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(text_surface, text_rect)
            
            score_surface = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_UI_TEXT)
            score_rect = score_surface.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(score_surface, score_rect)

    def _render_ui(self):
        # Health Bar
        bar_width, bar_height = 150, 20
        health_pct = self.player_health / self.PLAYER_HEALTH_MAX
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FILL, (10, 10, int(bar_width * health_pct), bar_height))

        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 35))

        # Timer
        timer_text = self.font_large.render(f"{self.game_timer:.1f}", True, self.COLOR_UI_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(timer_text, timer_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        self.reset()
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage to run and display the game
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for display ---
    pygame.display.set_caption("Zombie Survival")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        mov = 0 # no-op
        if keys[pygame.K_UP]: mov = 1
        elif keys[pygame.K_DOWN]: mov = 2
        elif keys[pygame.K_LEFT]: mov = 3
        elif keys[pygame.K_RIGHT]: mov = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov, space, shift]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        # The observation is already the rendered image, so we just blit it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for 'R' to reset
            while True:
                reset_event = pygame.event.wait()
                if reset_event.type == pygame.QUIT:
                    running = False
                    break
                if reset_event.type == pygame.KEYDOWN and reset_event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    break
            if not running:
                break

        clock.tick(env.FPS)
        
    env.close()