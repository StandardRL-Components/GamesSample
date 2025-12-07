
# Generated: 2025-08-28T04:14:17.352681
# Source Brief: brief_05184.md
# Brief Index: 5184

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move. Press space to fire. Press shift to start reloading."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a relentless zombie horde in a top-down arena shooter for 60 seconds."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_ARENA = (30, 35, 40)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_GLOW = (0, 100, 128, 64)
    COLOR_ZOMBIE_HEALTHY = (0, 150, 50)
    COLOR_ZOMBIE_DAMAGED = (180, 0, 0)
    COLOR_PROJECTILE = (255, 255, 255)
    COLOR_BLOOD = (200, 0, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_TIMER = (255, 50, 50)
    COLOR_HEALTH_BAR_BG = (80, 0, 0)
    COLOR_HEALTH_BAR_FG = (0, 200, 0)
    COLOR_AMMO_LOW = (255, 100, 0)

    # Player
    PLAYER_SIZE = 12
    PLAYER_SPEED = 4
    PLAYER_MAX_HEALTH = 100
    PLAYER_DAMAGE_FLASH_DURATION = 5

    # Weapon
    MAX_AMMO = 30
    FIRE_RATE = 4  # frames between shots
    RELOAD_TIME = 30  # frames
    PROJECTILE_SIZE = 4
    PROJECTILE_SPEED = 10

    # Zombies
    ZOMBIE_SIZE = 14
    ZOMBIE_SPEED = 1
    ZOMBIE_MAX_HEALTH = 5
    ZOMBIE_DAMAGE = 15
    INITIAL_ZOMBIE_SPAWN_INTERVAL = 25 # frames
    ZOMBIE_SPAWN_RATE_INCREASE = 0.005 # reduction per frame
    MIN_ZOMBIE_SPAWN_INTERVAL = 8

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_timer = pygame.font.Font(None, 36)
        self.font_reloading = pygame.font.Font(None, 20)

        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_damage_flash = None
        self.last_movement_dir = None
        self.ammo = None
        self.shot_cooldown = None
        self.reload_timer = None
        self.zombies = None
        self.projectiles = None
        self.particles = None
        self.game_timer = None
        self.zombie_spawn_timer = None
        self.zombie_spawn_interval = None
        self.steps = None
        self.score = None
        self.game_over = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = [self.WIDTH // 2, self.HEIGHT // 2]
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_damage_flash = 0
        self.last_movement_dir = np.array([0, -1]) # Default aim up
        
        self.ammo = self.MAX_AMMO
        self.shot_cooldown = 0
        self.reload_timer = 0

        self.zombies = []
        self.projectiles = []
        self.particles = []

        self.game_timer = self.GAME_DURATION_SECONDS
        self.zombie_spawn_interval = self.INITIAL_ZOMBIE_SPAWN_INTERVAL
        self.zombie_spawn_timer = self.zombie_spawn_interval

        self.steps = 0
        self.score = 0
        self.game_over = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.1  # Survival reward

        # --- Update Timers ---
        self.game_timer = max(0, self.game_timer - 1 / self.FPS)
        self.shot_cooldown = max(0, self.shot_cooldown - 1)
        self.zombie_spawn_timer = max(0, self.zombie_spawn_timer - 1)
        self.player_damage_flash = max(0, self.player_damage_flash - 1)

        if self.reload_timer > 0:
            self.reload_timer -= 1
            if self.reload_timer == 0:
                self.ammo = self.MAX_AMMO
                # sfx: reload_complete.wav

        # --- Handle Action ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement
        move_vec = np.array([0, 0])
        if movement == 1: move_vec[1] = -1 # Up
        elif movement == 2: move_vec[1] = 1  # Down
        elif movement == 3: move_vec[0] = -1 # Left
        elif movement == 4: move_vec[0] = 1  # Right
        
        if np.any(move_vec):
            self.last_movement_dir = move_vec
            self.player_pos[0] += move_vec[0] * self.PLAYER_SPEED
            self.player_pos[1] += move_vec[1] * self.PLAYER_SPEED

        # Clamp player position
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE // 2, self.WIDTH - self.PLAYER_SIZE // 2)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE // 2, self.HEIGHT - self.PLAYER_SIZE // 2)

        # Shooting
        if space_held and self.shot_cooldown == 0 and self.ammo > 0 and self.reload_timer == 0:
            self.projectiles.append({
                "pos": list(self.player_pos),
                "vel": self.last_movement_dir * self.PROJECTILE_SPEED
            })
            self.ammo -= 1
            self.shot_cooldown = self.FIRE_RATE
            # sfx: shoot.wav
            self._create_particles(self.player_pos, 1, (255, 255, 100), 5, 3, 2) # Muzzle flash

        # Reloading
        if shift_held and self.reload_timer == 0 and self.ammo < self.MAX_AMMO:
            self.reload_timer = self.RELOAD_TIME
            # sfx: reload_start.wav

        # --- Update Game Objects ---
        self._update_projectiles()
        self._update_zombies()
        self._update_particles()
        
        # --- Collisions ---
        reward += self._handle_collisions()

        # --- Spawning ---
        if self.zombie_spawn_timer == 0:
            self._spawn_zombie()
            self.zombie_spawn_interval = max(self.MIN_ZOMBIE_SPAWN_INTERVAL, self.zombie_spawn_interval - self.ZOMBIE_SPAWN_RATE_INCREASE)
            self.zombie_spawn_timer = self.zombie_spawn_interval

        # --- Termination Check ---
        terminated = False
        if self.player_health <= 0:
            reward = -100
            terminated = True
            # sfx: player_death.wav
        elif self.game_timer <= 0:
            reward += 50
            terminated = True
            # sfx: win.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        if terminated:
            self.game_over = True

        self.steps += 1
        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            if not (0 < p["pos"][0] < self.WIDTH and 0 < p["pos"][1] < self.HEIGHT):
                self.projectiles.remove(p)

    def _update_zombies(self):
        for z in self.zombies:
            direction = np.array(self.player_pos) - np.array(z["pos"])
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            z["pos"][0] += direction[0] * self.ZOMBIE_SPEED
            z["pos"][1] += direction[1] * self.ZOMBIE_SPEED
            
            # Pulse animation
            z["size_mod"] = math.sin(self.steps * 0.2 + z["offset"]) * 2

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            p["radius"] *= 0.95
            if p["life"] <= 0 or p["radius"] < 0.5:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        
        # Projectiles vs Zombies
        for p in self.projectiles[:]:
            proj_rect = pygame.Rect(p["pos"][0] - self.PROJECTILE_SIZE // 2, p["pos"][1] - self.PROJECTILE_SIZE // 2, self.PROJECTILE_SIZE, self.PROJECTILE_SIZE)
            for z in self.zombies[:]:
                zombie_rect = pygame.Rect(z["pos"][0] - self.ZOMBIE_SIZE // 2, z["pos"][1] - self.ZOMBIE_SIZE // 2, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
                if proj_rect.colliderect(zombie_rect):
                    z["health"] -= 1
                    self._create_particles(z["pos"], 10, self.COLOR_BLOOD, 10, 2, 4)
                    # sfx: zombie_hit.wav
                    if p in self.projectiles:
                        self.projectiles.remove(p)
                    
                    if z["health"] <= 0:
                        self.zombies.remove(z)
                        reward += 1
                        # sfx: zombie_death.wav
                    break # Projectile hits only one zombie

        # Zombies vs Player
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE // 2, self.player_pos[1] - self.PLAYER_SIZE // 2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        for z in self.zombies[:]:
            zombie_rect = pygame.Rect(z["pos"][0] - self.ZOMBIE_SIZE // 2, z["pos"][1] - self.ZOMBIE_SIZE // 2, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
            if player_rect.colliderect(zombie_rect):
                self.player_health = max(0, self.player_health - self.ZOMBIE_DAMAGE)
                self.player_damage_flash = self.PLAYER_DAMAGE_FLASH_DURATION
                self.zombies.remove(z)
                self._create_particles(self.player_pos, 20, self.COLOR_BLOOD, 15, 1, 3)
                # sfx: player_hit.wav
        return reward

    def _spawn_zombie(self):
        edge = random.randint(0, 3)
        if edge == 0: # Top
            pos = [random.randint(0, self.WIDTH), -self.ZOMBIE_SIZE]
        elif edge == 1: # Bottom
            pos = [random.randint(0, self.WIDTH), self.HEIGHT + self.ZOMBIE_SIZE]
        elif edge == 2: # Left
            pos = [-self.ZOMBIE_SIZE, random.randint(0, self.HEIGHT)]
        else: # Right
            pos = [self.WIDTH + self.ZOMBIE_SIZE, random.randint(0, self.HEIGHT)]
        
        self.zombies.append({
            "pos": pos,
            "health": self.ZOMBIE_MAX_HEALTH,
            "offset": random.uniform(0, 2 * math.pi),
            "size_mod": 0
        })

    def _create_particles(self, pos, count, color, life, min_speed, max_speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(min_speed, max_speed)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "radius": random.uniform(2, 5),
                "life": random.randint(life // 2, life),
                "color": color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        pygame.draw.rect(self.screen, self.COLOR_ARENA, (10, 10, self.WIDTH - 20, self.HEIGHT - 20))

        # Render particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p["color"], (int(p["pos"][0]), int(p["pos"][1])), int(p["radius"]))
            
        # Render zombies
        for z in self.zombies:
            health_ratio = z["health"] / self.ZOMBIE_MAX_HEALTH
            color = tuple(np.clip(np.array(self.COLOR_ZOMBIE_DAMAGED) * (1 - health_ratio) + np.array(self.COLOR_ZOMBIE_HEALTHY) * health_ratio, 0, 255).astype(int))
            size = int(self.ZOMBIE_SIZE + z["size_mod"])
            pygame.draw.rect(self.screen, color, (int(z["pos"][0] - size / 2), int(z["pos"][1] - size / 2), size, size))

        # Render projectiles
        for p in self.projectiles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, pos, self.PROJECTILE_SIZE // 2)

        # Render player
        if self.player_damage_flash > 0:
            player_color = self.COLOR_ZOMBIE_DAMAGED
        else:
            player_color = self.COLOR_PLAYER
        
        # Player glow
        glow_size = int(self.PLAYER_SIZE * 2.5)
        s = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(s, self.COLOR_PLAYER_GLOW, (glow_size//2, glow_size//2), glow_size//2)
        self.screen.blit(s, (int(self.player_pos[0] - glow_size/2), int(self.player_pos[1] - glow_size/2)))

        pygame.draw.rect(self.screen, player_color, (int(self.player_pos[0] - self.PLAYER_SIZE / 2), int(self.player_pos[1] - self.PLAYER_SIZE / 2), self.PLAYER_SIZE, self.PLAYER_SIZE))

        # Render damage flash overlay
        if self.player_damage_flash > 0:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((255, 0, 0, 40))
            self.screen.blit(s, (0,0))
            
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Health Bar
        bar_width, bar_height = 150, 15
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (20, 20, bar_width, bar_height))
        health_width = (self.player_health / self.PLAYER_MAX_HEALTH) * bar_width
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (20, 20, int(max(0, health_width)), bar_height))
        
        # Ammo Counter
        ammo_color = self.COLOR_UI_TEXT if self.ammo > 5 else self.COLOR_AMMO_LOW
        ammo_text = self.font_ui.render(f"AMMO: {self.ammo}/{self.MAX_AMMO}", True, ammo_color)
        self.screen.blit(ammo_text, (self.WIDTH - ammo_text.get_width() - 20, 20))
        
        # Reloading Indicator
        if self.reload_timer > 0:
            reload_text = self.font_reloading.render("RELOADING...", True, self.COLOR_AMMO_LOW)
            self.screen.blit(reload_text, (self.player_pos[0] - reload_text.get_width()//2, self.player_pos[1] - self.PLAYER_SIZE - 10))

        # Timer
        timer_text = self.font_timer.render(f"{math.ceil(self.game_timer)}", True, self.COLOR_TIMER)
        self.screen.blit(timer_text, (self.WIDTH // 2 - timer_text.get_width() // 2, 15))

        # Game Over Text
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,128))
            self.screen.blit(s, (0,0))
            
            font_gameover = pygame.font.Font(None, 72)
            if self.player_health <= 0:
                msg = "YOU DIED"
                color = self.COLOR_ZOMBIE_DAMAGED
            else:
                msg = "YOU SURVIVED"
                color = self.COLOR_PLAYER

            text_surface = font_gameover.render(msg, True, color)
            text_rect = text_surface.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surface, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "ammo": self.ammo,
            "timer": self.game_timer,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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


if __name__ == "__main__":
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    
    env = GameEnv()
    
    # --- Basic API test ---
    print("--- Running Basic API Test ---")
    obs, info = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    while not done and step_count < 2000:
        action = env.action_space.sample() # Random actions
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1
    print(f"API test finished after {step_count} steps. Final score: {total_reward:.2f}")
    print(f"Final info: {info}")
    
    # --- Interactive Test Setup ---
    # To run this, comment out the "os.environ" line above
    # and install pygame: pip install pygame
    
    # print("\n--- Starting Interactive Test ---")
    # env = GameEnv(render_mode="rgb_array")
    # obs, info = env.reset()
    # done = False
    # screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    # pygame.display.set_caption("Zombie Survival")
    # clock = pygame.time.Clock()
    
    # while not done:
    #     movement = 0 # No-op
    #     space_held = 0
    #     shift_held = 0
        
    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_UP]: movement = 1
    #     elif keys[pygame.K_DOWN]: movement = 2
    #     elif keys[pygame.K_LEFT]: movement = 3
    #     elif keys[pygame.K_RIGHT]: movement = 4
        
    #     if keys[pygame.K_SPACE]: space_held = 1
    #     if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
    #     action = [movement, space_held, shift_held]
        
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated
        
    #     # Display the observation from the environment
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
        
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             done = True
        
    #     clock.tick(GameEnv.FPS)
        
    # env.close()
    # print("Interactive test finished.")