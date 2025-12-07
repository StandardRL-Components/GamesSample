
# Generated: 2025-08-27T18:37:53.610601
# Source Brief: brief_01886.md
# Brief Index: 1886

        
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
        "Controls: Use arrow keys to aim the reticle. Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a zombie horde by strategically aiming and shooting from a fixed position in this side-view action game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.ZOMBIES_TO_WIN = 30
        self.PLAYER_MAX_HEALTH = 5

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GROUND = (40, 30, 30)
        self.COLOR_CITY_NEAR = (25, 25, 40)
        self.COLOR_CITY_FAR = (20, 20, 30)
        self.COLOR_PLAYER = (150, 200, 255)
        self.COLOR_RETICLE = (255, 255, 0, 200)
        self.COLOR_ZOMBIE_SKIN = (80, 120, 80)
        self.COLOR_ZOMBIE_HEAD = (90, 140, 90)
        self.COLOR_BULLET = (255, 255, 0)
        self.COLOR_BLOOD = (200, 0, 0)
        self.COLOR_MUZZLE_FLASH = (255, 200, 100)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HEALTH = (0, 200, 100)
        self.COLOR_DAMAGE = (255, 0, 0, 100)

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Player attributes
        self.player_pos = pygame.Vector2(80, self.HEIGHT - 80)
        self.reticle_speed = 15
        
        # Weapon attributes
        self.fire_cooldown = 5 # frames
        
        # Initialize state variables
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_health = self.PLAYER_MAX_HEALTH
        self.reticle_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        
        self.zombies = []
        self.bullets = []
        self.particles = []
        
        self.zombies_killed = 0
        self.zombies_spawned = 0
        
        self.zombie_spawn_timer = 0
        self.fire_cooldown_timer = 0
        self.muzzle_flash_timer = 0
        self.damage_flash_timer = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.1  # Cost of living
        
        # --- Handle Input ---
        self._handle_input(action)

        # --- Update Game State ---
        self._update_timers()
        self._update_bullets()
        self._update_zombies()
        self._spawn_zombies()
        self._update_particles()
        
        # --- Collisions and Rewards ---
        reward += self._handle_collisions()
        
        # --- Termination ---
        self.steps += 1
        terminated = self._check_termination()
        if terminated:
            if self.zombies_killed >= self.ZOMBIES_TO_WIN:
                reward += 100 # Win bonus
            else:
                reward += -100 # Loss penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # Reticle movement
        if movement == 1: self.reticle_pos.y -= self.reticle_speed
        elif movement == 2: self.reticle_pos.y += self.reticle_speed
        elif movement == 3: self.reticle_pos.x -= self.reticle_speed
        elif movement == 4: self.reticle_pos.x += self.reticle_speed
        
        self.reticle_pos.x = np.clip(self.reticle_pos.x, 0, self.WIDTH)
        self.reticle_pos.y = np.clip(self.reticle_pos.y, 0, self.HEIGHT)
        
        # Shooting
        if space_held and self.fire_cooldown_timer <= 0:
            # sfx: player_shoot.wav
            self.fire_cooldown_timer = self.fire_cooldown
            self.muzzle_flash_timer = 2 # frames
            
            direction = (self.reticle_pos - self.player_pos).normalize()
            
            bullet = {
                "pos": self.player_pos + direction * 40,
                "vel": direction * 25,
                "rect": pygame.Rect(0, 0, 6, 6)
            }
            self.bullets.append(bullet)

    def _update_timers(self):
        if self.fire_cooldown_timer > 0: self.fire_cooldown_timer -= 1
        if self.muzzle_flash_timer > 0: self.muzzle_flash_timer -= 1
        if self.damage_flash_timer > 0: self.damage_flash_timer -= 1
        if self.zombie_spawn_timer > 0: self.zombie_spawn_timer -= 1

    def _update_bullets(self):
        for bullet in self.bullets[:]:
            bullet["pos"] += bullet["vel"]
            bullet["rect"].center = bullet["pos"]
            if not self.screen.get_rect().colliderect(bullet["rect"]):
                self.bullets.remove(bullet)

    def _update_zombies(self):
        for zombie in self.zombies:
            zombie["pos"].x -= zombie["speed"]
            zombie["body_rect"].bottom = zombie["pos"].y
            zombie["body_rect"].centerx = zombie["pos"].x
            zombie["head_rect"].bottom = zombie["body_rect"].top
            zombie["head_rect"].centerx = zombie["pos"].x
            
            # Simple walk animation
            zombie["anim_timer"] = (zombie["anim_timer"] + 1) % 20

    def _spawn_zombies(self):
        if self.zombie_spawn_timer <= 0 and self.zombies_spawned < self.ZOMBIES_TO_WIN:
            # sfx: zombie_groan.wav
            self.zombies_spawned += 1
            
            # Difficulty scaling
            spawn_delay = 2.0
            if self.zombies_killed >= 10: spawn_delay = 1.5
            if self.zombies_killed >= 20: spawn_delay = 1.0
            self.zombie_spawn_timer = int(spawn_delay * self.FPS * (self.np_random.random() * 0.5 + 0.75))

            z_height = self.np_random.integers(50, 81)
            z_width = int(z_height / 2.5)
            
            zombie = {
                "pos": pygame.Vector2(self.WIDTH + z_width, self.HEIGHT - 30),
                "speed": self.np_random.uniform(1.0, 2.5),
                "health": 2,
                "height": z_height,
                "width": z_width,
                "body_rect": pygame.Rect(0, 0, z_width, z_height - int(z_height/4)),
                "head_rect": pygame.Rect(0, 0, int(z_width*0.8), int(z_height/4)),
                "anim_timer": 0,
            }
            self.zombies.append(zombie)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"].y += 0.2 # gravity
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        
        # Bullets vs Zombies
        for bullet in self.bullets[:]:
            for zombie in self.zombies[:]:
                hit = False
                # Check headshot first
                if zombie["head_rect"].colliderect(bullet["rect"]):
                    hit = True
                    is_headshot = True
                    damage = 2
                    reward += 1.0
                elif zombie["body_rect"].colliderect(bullet["rect"]):
                    hit = True
                    is_headshot = False
                    damage = 1
                    reward += 0.5

                if hit:
                    # sfx: bullet_impact.wav
                    zombie["health"] -= damage
                    self._create_blood_splatter(bullet["pos"])
                    if bullet in self.bullets: self.bullets.remove(bullet)
                    
                    if zombie["health"] <= 0:
                        # sfx: zombie_die.wav
                        kill_reward = 20 if is_headshot else 10
                        reward += kill_reward
                        self.score += kill_reward
                        self.zombies_killed += 1
                        self.zombies.remove(zombie)
                    break # Bullet can only hit one zombie
        
        # Zombies vs Player
        player_hitbox = pygame.Rect(self.player_pos.x - 20, self.player_pos.y - 60, 40, 60)
        for zombie in self.zombies[:]:
            if zombie["body_rect"].colliderect(player_hitbox):
                # sfx: player_hurt.wav
                self.player_health -= 1
                self.damage_flash_timer = 5
                self.zombies.remove(zombie)

        return reward

    def _create_blood_splatter(self, pos):
        for _ in range(self.np_random.integers(10, 20)):
            particle = {
                "pos": pygame.Vector2(pos),
                "vel": pygame.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-4, 1)),
                "life": self.np_random.integers(15, 30),
                "radius": self.np_random.uniform(2, 5)
            }
            self.particles.append(particle)

    def _check_termination(self):
        if self.player_health <= 0:
            self.game_over = True
        if self.zombies_killed >= self.ZOMBIES_TO_WIN:
            self.game_over = True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        return self.game_over
    
    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "zombies_killed": self.zombies_killed,
            "zombies_remaining": self.ZOMBIES_TO_WIN - self.zombies_killed
        }

    def _render_game(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        # Cityscape
        for _ in range(5):
            pygame.draw.rect(self.screen, self.COLOR_CITY_FAR, (random.randint(0, self.WIDTH), self.HEIGHT - 100 - random.randint(0, 50), random.randint(20, 80), 100))
        for _ in range(3):
            pygame.draw.rect(self.screen, self.COLOR_CITY_NEAR, (random.randint(0, self.WIDTH), self.HEIGHT - 80 - random.randint(0, 30), random.randint(40, 100), 100))
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.HEIGHT - 30, self.WIDTH, 30))

        # --- Player ---
        player_rect = pygame.Rect(self.player_pos.x - 15, self.player_pos.y - 60, 30, 60)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        # Gun
        gun_angle = math.atan2(self.reticle_pos.y - self.player_pos.y, self.reticle_pos.x - self.player_pos.x)
        gun_end_x = self.player_pos.x + 40 * math.cos(gun_angle)
        gun_end_y = self.player_pos.y + 40 * math.sin(gun_angle)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, self.player_pos, (gun_end_x, gun_end_y), 8)
        
        # --- Zombies ---
        for z in self.zombies:
            # Body
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE_SKIN, z["body_rect"])
            # Head
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE_HEAD, z["head_rect"])
            # Leg animation
            leg_offset = 5 if z["anim_timer"] < 10 else -5
            leg1 = pygame.Rect(z["body_rect"].left + 5, z["body_rect"].bottom, 8, 15)
            leg2 = pygame.Rect(z["body_rect"].right - 13, z["body_rect"].bottom, 8, 15)
            leg1.x += leg_offset
            leg2.x -= leg_offset
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE_SKIN, leg1)
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE_SKIN, leg2)

        # --- Bullets ---
        for b in self.bullets:
            pygame.draw.circle(self.screen, self.COLOR_BULLET, b["pos"], 3)
            
        # --- Effects ---
        # Blood particles
        for p in self.particles:
            pygame.draw.circle(self.screen, self.COLOR_BLOOD, p["pos"], int(p["radius"] * (p["life"] / 30.0)))
        
        # Muzzle flash
        if self.muzzle_flash_timer > 0:
            flash_pos = (gun_end_x, gun_end_y)
            pygame.draw.circle(self.screen, self.COLOR_MUZZLE_FLASH, flash_pos, 15)
            pygame.draw.circle(self.screen, (255, 255, 255), flash_pos, 8)
            
        # Damage flash
        if self.damage_flash_timer > 0:
            damage_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            damage_surface.fill(self.COLOR_DAMAGE)
            self.screen.blit(damage_surface, (0, 0))

        # --- UI ---
        # Reticle
        pygame.gfxdraw.aacircle(self.screen, int(self.reticle_pos.x), int(self.reticle_pos.y), 12, self.COLOR_RETICLE)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (self.reticle_pos.x - 20, self.reticle_pos.y), (self.reticle_pos.x - 5, self.reticle_pos.y), 1)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (self.reticle_pos.x + 5, self.reticle_pos.y), (self.reticle_pos.x + 20, self.reticle_pos.y), 1)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (self.reticle_pos.x, self.reticle_pos.y - 20), (self.reticle_pos.x, self.reticle_pos.y - 5), 1)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (self.reticle_pos.x, self.reticle_pos.y + 5), (self.reticle_pos.x, self.reticle_pos.y + 20), 1)
        
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Health
        health_text = self.font_small.render("HEALTH:", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.WIDTH - 180, 15))
        for i in range(self.PLAYER_MAX_HEALTH):
            color = self.COLOR_HEALTH if i < self.player_health else self.COLOR_GROUND
            pygame.draw.rect(self.screen, color, (self.WIDTH - 100 + i * 20, 15, 15, 15))
            
        # Zombies remaining
        zombie_count_text = self.font_small.render(f"TARGETS: {self.ZOMBIES_TO_WIN - self.zombies_killed}", True, self.COLOR_TEXT)
        self.screen.blit(zombie_count_text, (10, self.HEIGHT - 30))
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0,0))
            
            win_condition = self.zombies_killed >= self.ZOMBIES_TO_WIN
            msg = "MISSION COMPLETE" if win_condition else "GAME OVER"
            color = self.COLOR_HEALTH if win_condition else self.COLOR_BLOOD
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- To display the game in a window ---
    # Note: This is for human testing and visualization, not for training.
    # The environment itself is headless as required.
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Zombie Horde")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    # Map pygame keys to the MultiDiscrete action space
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    running = True
    while running:
        # --- Human Input ---
        movement_action = 0 # no-op
        space_action = 0 # released
        shift_action = 0 # released (unused)
        
        keys = pygame.key.get_pressed()
        
        for key, action_val in key_map.items():
            if keys[key]:
                movement_action = action_val
                break # Only one movement key at a time
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        action = [movement_action, space_action, shift_action]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}. Press 'R' to restart.")

        clock.tick(env.FPS)
        
    pygame.quit()