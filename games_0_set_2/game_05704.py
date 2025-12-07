
# Generated: 2025-08-28T05:49:17.051290
# Source Brief: brief_05704.md
# Brief Index: 5704

        
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

    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Press space to fire your weapon."
    )

    game_description = (
        "Survive hordes of zombies in a side-scrolling cityscape. Clear 5 waves to win."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GROUND_Y = 350
    FPS = 30

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GROUND = (50, 50, 60)
    COLOR_PLAYER = (100, 255, 100)
    COLOR_PLAYER_DMG = (255, 100, 100)
    COLOR_ZOMBIE = (100, 120, 80)
    COLOR_BULLET = (255, 200, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_HEALTH = (0, 200, 0)
    COLOR_HEALTH_BG = (200, 0, 0)
    COLOR_AMMO = (200, 200, 0)
    COLOR_GAMEOVER = (200, 30, 30)
    COLOR_WIN = (30, 200, 30)

    # Player Physics
    PLAYER_SPEED = 5
    PLAYER_JUMP_POWER = -13
    GRAVITY = 0.7
    PLAYER_HEALTH_MAX = 100
    PLAYER_DMG_COOLDOWN = 30  # frames

    # Weapon
    BULLET_SPEED = 15
    SHOOT_COOLDOWN = 8  # frames
    INITIAL_AMMO = 50

    # Waves
    WAVE_ZOMBIE_COUNTS = [0, 20, 25, 30, 35, 40] # Index 0 is unused
    WAVE_ZOMBIE_SPEEDS = [0, 1.0, 1.1, 1.2, 1.3, 1.4]
    MAX_WAVES = 5

    # Episode
    MAX_STEPS = 5000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Initialize state variables to be populated in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_game = False
        self.player = None
        self.zombies = []
        self.bullets = []
        self.particles = []
        self.cityscape = []
        self.current_wave = 0
        self.zombies_to_spawn_this_wave = 0
        self.zombie_spawn_timer = 0
        self.ammo = 0
        self.shoot_cooldown_timer = 0
        self.player_dmg_timer = 0
        self.rng = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_game = False

        self.player = {
            "rect": pygame.Rect(self.SCREEN_WIDTH // 2, self.GROUND_Y - 40, 20, 40),
            "vel": pygame.Vector2(0, 0),
            "on_ground": False,
            "facing": 1,  # 1 for right, -1 for left
            "health": self.PLAYER_HEALTH_MAX,
        }
        
        self.zombies = []
        self.bullets = []
        self.particles = []
        self.current_wave = 0
        self.ammo = self.INITIAL_AMMO
        self.shoot_cooldown_timer = 0
        self.player_dmg_timer = 0
        
        self._procedurally_generate_cityscape()
        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0.1  # Survival reward per frame

        if not self.game_over:
            self._handle_input(action)
            self._update_player()
            self._update_zombies()
            self._update_bullets()
            
            kill_reward = self._handle_collisions()
            reward += kill_reward
            
            wave_reward, wave_cleared = self._check_wave_completion()
            if wave_cleared:
                reward += wave_reward
                if self.current_wave > self.MAX_WAVES:
                    self.win_game = True
                else:
                    self._start_next_wave()
        
        self._update_particles()

        self.steps += 1
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        
        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Movement
        if movement == 3:  # Left
            self.player["vel"].x = -self.PLAYER_SPEED
            self.player["facing"] = -1
        elif movement == 4:  # Right
            self.player["vel"].x = self.PLAYER_SPEED
            self.player["facing"] = 1
        else:
            self.player["vel"].x = 0

        # Jump
        if movement == 1 and self.player["on_ground"]:
            self.player["vel"].y = self.PLAYER_JUMP_POWER
            self.player["on_ground"] = False
            # sfx: jump

        # Shoot
        if space_held and self.shoot_cooldown_timer == 0 and self.ammo > 0:
            self._fire_bullet()
            self.shoot_cooldown_timer = self.SHOOT_COOLDOWN
            self.ammo -= 1
            # sfx: shoot
    
    def _update_player(self):
        # Update timers
        if self.shoot_cooldown_timer > 0:
            self.shoot_cooldown_timer -= 1
        if self.player_dmg_timer > 0:
            self.player_dmg_timer -= 1

        # Apply gravity
        self.player["vel"].y += self.GRAVITY
        
        # Move player
        self.player["rect"].x += self.player["vel"].x
        self.player["rect"].y += self.player["vel"].y

        # Horizontal wrapping
        if self.player["rect"].right < 0:
            self.player["rect"].left = self.SCREEN_WIDTH
        elif self.player["rect"].left > self.SCREEN_WIDTH:
            self.player["rect"].right = 0

        # Ground collision
        if self.player["rect"].bottom >= self.GROUND_Y:
            self.player["rect"].bottom = self.GROUND_Y
            self.player["vel"].y = 0
            self.player["on_ground"] = True
        else:
            self.player["on_ground"] = False

    def _update_zombies(self):
        # Spawn new zombies if needed
        self.zombie_spawn_timer -= 1
        if self.zombies_to_spawn_this_wave > 0 and self.zombie_spawn_timer <= 0:
            self._spawn_zombie()
            self.zombies_to_spawn_this_wave -= 1
            self.zombie_spawn_timer = self.rng.integers(30, 90) # Spawn every 1-3 seconds

        # Move existing zombies
        speed = self.WAVE_ZOMBIE_SPEEDS[self.current_wave]
        for z in self.zombies:
            if self.player["rect"].centerx < z["rect"].centerx:
                z["rect"].x -= speed
            else:
                z["rect"].x += speed
            
            # Horizontal wrapping
            if z["rect"].right < 0:
                z["rect"].left = self.SCREEN_WIDTH
            elif z["rect"].left > self.SCREEN_WIDTH:
                z["rect"].right = 0

    def _update_bullets(self):
        for b in self.bullets[:]:
            b["rect"].x += b["vel"].x
            if not self.screen.get_rect().colliderect(b["rect"]):
                self.bullets.remove(b)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        kill_reward = 0

        # Player-Zombie collision
        if self.player_dmg_timer == 0:
            for z in self.zombies:
                if self.player["rect"].colliderect(z["rect"]):
                    self.player["health"] = max(0, self.player["health"] - 10)
                    self.player_dmg_timer = self.PLAYER_DMG_COOLDOWN
                    # sfx: player_hurt
                    self._create_particles(self.player["rect"].center, 10, self.COLOR_PLAYER_DMG)
                    break 

        # Bullet-Zombie collision
        for b in self.bullets[:]:
            for z in self.zombies[:]:
                if b["rect"].colliderect(z["rect"]):
                    self.zombies.remove(z)
                    if b in self.bullets: self.bullets.remove(b)
                    self.score += 10
                    kill_reward += 1
                    # sfx: zombie_die
                    self._create_particles(z["rect"].center, 20, self.COLOR_ZOMBIE)
                    break
            else:
                continue
            break
        
        return kill_reward

    def _check_wave_completion(self):
        if self.zombies_to_spawn_this_wave == 0 and not self.zombies:
            if 0 < self.current_wave <= self.MAX_WAVES:
                self.score += 100
                return 100, True # reward, wave_cleared
        return 0, False

    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave <= self.MAX_WAVES:
            self.zombies_to_spawn_this_wave = self.WAVE_ZOMBIE_COUNTS[self.current_wave]
            self.zombie_spawn_timer = 60 # Initial delay
            self.ammo = self.INITIAL_AMMO # Replenish ammo

    def _spawn_zombie(self):
        side = self.rng.choice([-1, 1])
        x = -30 if side == -1 else self.SCREEN_WIDTH + 30
        self.zombies.append({
            "rect": pygame.Rect(x, self.GROUND_Y - 35, 25, 35)
        })

    def _fire_bullet(self):
        start_pos_x = self.player["rect"].right if self.player["facing"] == 1 else self.player["rect"].left
        start_pos_y = self.player["rect"].centery - 5
        bullet_rect = pygame.Rect(start_pos_x, start_pos_y, 8, 4)
        bullet_vel = pygame.Vector2(self.BULLET_SPEED * self.player["facing"], 0)
        self.bullets.append({"rect": bullet_rect, "vel": bullet_vel})
        
        # Muzzle flash
        flash_pos = pygame.Vector2(bullet_rect.center)
        self._create_particles(flash_pos, 5, self.COLOR_BULLET, life=5, speed_mult=3)
    
    def _create_particles(self, pos, count, color, life=15, speed_mult=2):
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 3) * speed_mult
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "lifetime": self.rng.integers(life // 2, life),
                "color": color,
                "size": self.rng.integers(2, 5)
            })

    def _check_termination(self):
        if self.player["health"] <= 0:
            return True, -100 # terminated, reward
        if self.win_game:
            return True, 500
        if self.steps >= self.MAX_STEPS:
            return True, 0
        return False, 0

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player["health"],
            "wave": self.current_wave,
            "ammo": self.ammo
        }

    def _procedurally_generate_cityscape(self):
        self.cityscape.clear()
        x = -50
        while x < self.SCREEN_WIDTH + 50:
            w = self.rng.integers(40, 100)
            h = self.rng.integers(50, 250)
            y = self.GROUND_Y - h
            color_val = self.rng.integers(35, 55)
            color = (color_val, color_val, color_val + 10)
            self.cityscape.append({"rect": pygame.Rect(x, y, w, h), "color": color})
            x += w + self.rng.integers(10, 30)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_cityscape()
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))
        self._render_zombies()
        self._render_player()
        self._render_bullets()
        self._render_particles()
        self._render_ui()
        
        if self.game_over:
            self._render_end_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_cityscape(self):
        for building in self.cityscape:
            pygame.draw.rect(self.screen, building["color"], building["rect"])

    def _render_player(self):
        color = self.COLOR_PLAYER
        if self.player_dmg_timer > 0 and (self.steps // 3) % 2 == 0:
            color = self.COLOR_PLAYER_DMG
        
        pygame.draw.rect(self.screen, color, self.player["rect"])
        
        # Eye to show direction
        eye_x = self.player["rect"].centerx + (self.player["facing"] * 5)
        eye_y = self.player["rect"].top + 10
        pygame.draw.rect(self.screen, self.COLOR_BG, (eye_x, eye_y, 3, 3))

    def _render_zombies(self):
        for z in self.zombies:
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, z["rect"])

    def _render_bullets(self):
        for b in self.bullets:
            pygame.draw.rect(self.screen, self.COLOR_BULLET, b["rect"])

    def _render_particles(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, p["color"], p["pos"], p["size"] * (p["lifetime"] / 15.0))

    def _render_ui(self):
        # Health Bar
        health_pct = self.player["health"] / self.PLAYER_HEALTH_MAX
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, (10, 10, 200 * health_pct, 20))

        # Wave Counter
        wave_text = self.font_small.render(f"WAVE: {min(self.current_wave, self.MAX_WAVES)}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))

        # Ammo
        ammo_text = self.font_small.render(f"AMMO: {self.ammo}", True, self.COLOR_AMMO)
        self.screen.blit(ammo_text, (10, 35))

        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, self.SCREEN_HEIGHT - 25))

    def _render_end_screen(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        if self.win_game:
            text = self.font_large.render("YOU SURVIVED", True, self.COLOR_WIN)
        else:
            text = self.font_large.render("GAME OVER", True, self.COLOR_GAMEOVER)
        
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
        self.screen.blit(text, text_rect)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Zombie Survival")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Set SDL_VIDEODRIVER to "dummy" if you're running headless
    # import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        if keys[pygame.K_r]: # Press R to reset
             obs, info = env.reset()
             total_reward = 0
             print("--- GAME RESET ---")

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Transpose back for pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(GameEnv.FPS)

    env.close()