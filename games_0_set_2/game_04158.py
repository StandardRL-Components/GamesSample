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

    # Must be a short, user-facing control string:
    user_guide = "Controls: ←→ to move, ↑↓ to aim. Press space to shoot. Hold shift to reload."

    # Must be a short, user-facing description of the game:
    game_description = "Survive waves of zombies in a fast-paced, side-scrolling arcade shooter."

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.FPS = 30

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_BG_BUILDING = (30, 35, 50)
        self.COLOR_PLAYER = (50, 200, 100)
        self.COLOR_GUN = (180, 180, 180)
        self.COLOR_ZOMBIE = (220, 50, 50)
        self.COLOR_PROJECTILE = (255, 255, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_HEALTH_BAR_BG = (100, 0, 0)
        self.COLOR_HEALTH_BAR_FG = (0, 200, 0)
        self.COLOR_AMMO_BAR_BG = (80, 80, 80)
        self.COLOR_AMMO_BAR_FG = (100, 150, 255)

        # Player settings
        self.PLAYER_SPEED = 6
        self.PLAYER_HEALTH_MAX = 100
        self.PLAYER_AIM_SPEED = 3 # degrees per frame
        self.PLAYER_AIM_LIMIT = 45 # degrees
        
        # Weapon settings
        self.MAX_AMMO = 12
        self.SHOOT_COOLDOWN_FRAMES = 5 # 6 shots per second
        self.RELOAD_TIME_FRAMES = 60 # 2 seconds
        self.PROJECTILE_SPEED = 15

        # Zombie settings
        self.BASE_ZOMBIE_COUNT = 3
        self.BASE_ZOMBIE_SPEED = 1.0
        self.BASE_ZOMBIE_HEALTH = 3
        self.ZOMBIE_DAMAGE = 10
        
        # Wave settings
        self.WAVE_CLEAR_DELAY = 90 # 3 seconds

        # Reward settings
        self.REWARD_HIT_ZOMBIE = 0.1
        self.REWARD_MISS_SHOT = -0.01
        self.REWARD_KILL_ZOMBIE = 1.0
        self.REWARD_WAVE_CLEAR = 10.0
        self.REWARD_GAME_OVER = -10.0

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 36)
        self.font_l = pygame.font.Font(None, 72)
        
        # Internal state variables are initialized in reset()
        self.np_random = None
        self.player_pos = None
        self.player_health = None
        self.player_aim_angle = None
        self.player_rect = None
        self.ammo = None
        self.shoot_cooldown = None
        self.reloading_timer = None
        self.zombies = None
        self.projectiles = None
        self.particles = None
        self.wave = None
        self.wave_clear_timer = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.background_buildings = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        # Player state
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 40)
        self.player_rect = pygame.Rect(0, 0, 20, 40)
        self.player_rect.center = self.player_pos
        self.player_health = self.PLAYER_HEALTH_MAX
        self.player_aim_angle = 0
        
        # Weapon state
        self.ammo = self.MAX_AMMO
        self.shoot_cooldown = 0
        self.reloading_timer = 0
        
        # Game state
        self.zombies = []
        self.projectiles = []
        self.particles = []
        self.wave = 1
        self.wave_clear_timer = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self._generate_background()
        self._start_new_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.clock.tick(self.FPS)
        
        reward = 0
        terminated = False
        truncated = False

        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        if self.wave_clear_timer > 0:
            self.wave_clear_timer -= 1
            if self.wave_clear_timer == 0:
                self.wave += 1
                self._start_new_wave()
                reward += self.REWARD_WAVE_CLEAR
        else:
            self._handle_input(action)
            self._update_player_state()
            
            projectile_rewards, zombie_kill_rewards = self._update_world()
            reward += projectile_rewards + zombie_kill_rewards

            if not self.zombies and self.wave_clear_timer == 0: # Check timer to avoid re-triggering
                self.wave_clear_timer = self.WAVE_CLEAR_DELAY
        
        self.steps += 1
        if self.player_health <= 0:
            self.game_over = True
            terminated = True
            reward += self.REWARD_GAME_OVER
        elif self.steps >= self.MAX_STEPS:
            truncated = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Aiming
        if movement == 1: # Up
            self.player_aim_angle = min(self.PLAYER_AIM_LIMIT, self.player_aim_angle + self.PLAYER_AIM_SPEED)
        elif movement == 2: # Down
            self.player_aim_angle = max(-self.PLAYER_AIM_LIMIT, self.player_aim_angle - self.PLAYER_AIM_SPEED)
        elif movement == 0: # None, return to center
            if self.player_aim_angle > 0:
                self.player_aim_angle = max(0, self.player_aim_angle - self.PLAYER_AIM_SPEED)
            elif self.player_aim_angle < 0:
                self.player_aim_angle = min(0, self.player_aim_angle + self.PLAYER_AIM_SPEED)
        
        # Movement
        if movement == 3: # Left
            self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos.x += self.PLAYER_SPEED
        self.player_pos.x = np.clip(self.player_pos.x, self.player_rect.width / 2, self.WIDTH - self.player_rect.width / 2)
        self.player_rect.centerx = int(self.player_pos.x)

        # Shooting
        if space_held and self.shoot_cooldown == 0 and self.ammo > 0 and self.reloading_timer == 0:
            self._spawn_projectile()
            self.ammo -= 1
            self.shoot_cooldown = self.SHOOT_COOLDOWN_FRAMES

        # Reloading
        if shift_held and self.ammo < self.MAX_AMMO and self.reloading_timer == 0:
            self.reloading_timer = self.RELOAD_TIME_FRAMES

    def _update_player_state(self):
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        if self.reloading_timer > 0:
            self.reloading_timer -= 1
            if self.reloading_timer == 0:
                self.ammo = self.MAX_AMMO

    def _update_world(self):
        projectile_rewards = self._update_projectiles()
        zombie_kill_rewards = self._update_zombies()
        self._update_particles()
        return projectile_rewards, zombie_kill_rewards

    def _start_new_wave(self):
        num_zombies = self.BASE_ZOMBIE_COUNT + self.wave - 1
        zombie_speed = self.BASE_ZOMBIE_SPEED + (self.wave - 1) * 0.05
        zombie_health = self.BASE_ZOMBIE_HEALTH + int((self.wave - 1) / 2)

        for _ in range(num_zombies):
            side = self.np_random.choice([-1, 1])
            spawn_x = -30 if side == -1 else self.WIDTH + 30
            spawn_y = self.HEIGHT - 35
            self.zombies.append({
                "rect": pygame.Rect(spawn_x, spawn_y, 30, 35),
                "health": zombie_health,
                "speed": zombie_speed,
                "vel": pygame.Vector2(0, 0)
            })

    def _spawn_projectile(self):
        angle_rad = math.radians(-self.player_aim_angle)
        start_pos = self.player_pos + pygame.Vector2(0, -20)
        vel = pygame.Vector2(math.cos(angle_rad), math.sin(angle_rad)) * self.PROJECTILE_SPEED
        self.projectiles.append({"pos": pygame.Vector2(start_pos), "vel": vel})
        self._create_particles(start_pos, 5, self.COLOR_PROJECTILE, 1, 3, 5) # Muzzle flash

    def _update_projectiles(self):
        reward = 0
        for p in self.projectiles[:]:
            p["pos"] += p["vel"]
            
            hit_zombie = False
            for z in self.zombies[:]:
                if z["rect"].collidepoint(p["pos"]):
                    z["health"] -= 1
                    reward += self.REWARD_HIT_ZOMBIE
                    self._create_particles(p["pos"], 10, (255, 150, 50), 2, 4, 10) # Hit spark
                    self.projectiles.remove(p)
                    hit_zombie = True
                    break
            
            if hit_zombie:
                continue

            if not (0 < p["pos"].x < self.WIDTH and 0 < p["pos"].y < self.HEIGHT):
                if p in self.projectiles:
                    self.projectiles.remove(p)
                    reward += self.REWARD_MISS_SHOT
        return reward

    def _update_zombies(self):
        reward = 0
        for z in self.zombies[:]:
            if z["health"] <= 0:
                self.score += 1
                reward += self.REWARD_KILL_ZOMBIE
                self._create_particles(pygame.Vector2(z["rect"].center), 30, self.COLOR_ZOMBIE, 3, 6, 20)
                self.zombies.remove(z)
                continue

            direction = (self.player_pos - pygame.Vector2(z["rect"].center)).normalize()
            z["vel"] = direction * z["speed"]
            z["rect"].x += z["vel"].x
            z["rect"].y += z["vel"].y

            if self.player_rect.colliderect(z["rect"]):
                self.player_health -= self.ZOMBIE_DAMAGE
                self.player_health = max(0, self.player_health)
                self._create_particles(self.player_pos, 15, self.COLOR_PLAYER, 2, 5, 15)
                z["rect"].x -= direction.x * 20
        return reward
    
    def _create_particles(self, pos, count, color, min_speed, max_speed, life):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({"pos": pygame.Vector2(pos), "vel": vel, "life": life, "max_life": life, "color": color})
    
    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.9 # friction
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _generate_background(self):
        self.background_buildings.clear()
        for i in range(20):
            w = self.np_random.integers(30, 80)
            h = self.np_random.integers(50, 250)
            x = self.np_random.integers(-20, self.WIDTH)
            y = self.HEIGHT - h
            self.background_buildings.append(pygame.Rect(x, y, w, h))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Background
        for building in self.background_buildings:
            pygame.draw.rect(self.screen, self.COLOR_BG_BUILDING, building)
        pygame.draw.rect(self.screen, (10, 10, 15), (0, self.HEIGHT - 20, self.WIDTH, 20))

        # Particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = (*p["color"], alpha)
            size = max(1, int(3 * (p["life"] / p["max_life"])))
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (int(p["pos"].x - size), int(p["pos"].y - size)))

        # Zombies
        for z in self.zombies:
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, z["rect"])
            pygame.draw.rect(self.screen, (0,0,0), z["rect"], 1) # Outline

        # Player
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect, border_radius=4)
        
        # Player Gun
        gun_surf = pygame.Surface((30, 6), pygame.SRCALPHA)
        gun_surf.fill(self.COLOR_GUN)
        rotated_gun = pygame.transform.rotate(gun_surf, self.player_aim_angle)
        gun_rect = rotated_gun.get_rect(center=self.player_pos + pygame.Vector2(0, -15))
        self.screen.blit(rotated_gun, gun_rect)

        # Projectiles
        for p in self.projectiles:
            pos = (int(p["pos"].x), int(p["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / self.PLAYER_HEALTH_MAX)
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (10, 10, int(bar_width * health_ratio), 20))
        health_text = self.font_s.render(f"HP", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Ammo Bar
        ammo_ratio = self.ammo / self.MAX_AMMO
        if self.reloading_timer > 0:
            ammo_ratio = (self.RELOAD_TIME_FRAMES - self.reloading_timer) / self.RELOAD_TIME_FRAMES
        
        bar_width = 120
        pygame.draw.rect(self.screen, self.COLOR_AMMO_BAR_BG, (10, 35, bar_width, 15))
        pygame.draw.rect(self.screen, self.COLOR_AMMO_BAR_FG, (10, 35, int(bar_width * ammo_ratio), 15))
        ammo_text_str = f"AMMO: {self.ammo}/{self.MAX_AMMO}"
        if self.reloading_timer > 0:
            ammo_text_str = "RELOADING"
        ammo_text = self.font_s.render(ammo_text_str, True, self.COLOR_TEXT)
        self.screen.blit(ammo_text, (15, 35))

        # Score
        score_text = self.font_m.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

        # Wave
        wave_text = self.font_m.render(f"WAVE: {self.wave}", True, self.COLOR_TEXT)
        wave_rect = wave_text.get_rect(midbottom=(self.WIDTH / 2, self.HEIGHT - 5))
        self.screen.blit(wave_text, wave_rect)

        # Status Text
        if self.wave_clear_timer > 0 and not self.game_over:
            status_text = self.font_l.render("WAVE CLEARED", True, self.COLOR_HEALTH_BAR_FG)
            status_rect = status_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(status_text, status_rect)
        elif self.game_over:
            status_text = self.font_l.render("GAME OVER", True, self.COLOR_ZOMBIE)
            status_rect = status_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(status_text, status_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "health": self.player_health,
            "ammo": self.ammo,
        }
    
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Make sure to handle the case where the display driver is 'dummy'
    is_headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"
    
    env = GameEnv()
    obs, info = env.reset()
    
    if not is_headless:
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (50, 50) # Position window
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Zombie Wave Survivor")
    
    terminated = False
    truncated = False
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0
    
    print(env.user_guide)

    while not terminated and not truncated:
        if not is_headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    terminated = True
            
            # Player controls
            keys = pygame.key.get_pressed()
            
            if keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            elif keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            else: movement = 0
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
        else:
            # In headless mode, sample actions
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        
        if not is_headless:
            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(env.FPS) # Control the frame rate for playability

    env.close()
    print(f"Game Over! Final Score: {info['score']}, Wave: {info['wave']}")