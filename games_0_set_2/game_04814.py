
# Generated: 2025-08-28T03:05:22.066069
# Source Brief: brief_04814.md
# Brief Index: 4814

        
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
        "Controls: ↑↓ to move. ←→ to aim. Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive hordes of zombies in a side-scrolling shooter. "
        "Grab ammo crates and hold out until the timer runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GROUND_LEVEL = self.HEIGHT - 60
        self.MAX_STEPS = 3000

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GROUND = (40, 35, 30)
        self.COLOR_CITY = (30, 35, 50)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_ZOMBIE = (200, 50, 50)
        self.COLOR_AMMO = (100, 255, 100)
        self.COLOR_PROJECTILE = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR = (0, 200, 0)
        self.COLOR_HEALTH_BAR_BG = (100, 0, 0)
        self.COLOR_MUZZLE_FLASH = (255, 255, 150)

        # Game Parameters
        self.PLAYER_SPEED = 4
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_MAX_AMMO = 100
        self.PLAYER_SHOOT_COOLDOWN = 5
        self.ZOMBIE_HEALTH = 20
        self.ZOMBIE_DAMAGE = 10
        self.PROJECTILE_SPEED = 12
        self.PROJECTILE_DAMAGE = 20
        self.AMMO_REFILL = 25

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 72)
        
        # Initialize state variables
        self.city_scape = [] # Persists across resets for consistent background
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_rect = pygame.Rect(100, self.GROUND_LEVEL - 40, 20, 40)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_ammo = 50
        self.player_facing_right = True
        self.player_hit_timer = 0
        self.shoot_cooldown_timer = 0
        self.prev_space_held = False

        self.zombies = []
        self.projectiles = []
        self.ammo_crates = []
        self.particles = []

        self.zombie_spawn_timer = 0
        self.ammo_spawn_timer = 100

        if not self.city_scape:
            self._generate_background()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.1  # Survival reward

        self._handle_input(action)
        self._update_timers()
        
        self._update_projectiles()
        zombie_kill_reward, player_damage = self._update_zombies()
        ammo_pickup_reward = self._update_ammo_crates()
        self._update_particles()
        
        if player_damage > 0:
            self.player_health -= player_damage
            self.player_hit_timer = 10 # Flash for 10 frames
            # sound: player_hurt.wav
        
        self._spawn_zombies()
        self._spawn_ammo_crates()
        
        reward += zombie_kill_reward + ammo_pickup_reward

        terminated = self.player_health <= 0 or self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True
            if self.player_health <= 0:
                reward = -10.0  # Death penalty
            elif self.steps >= self.MAX_STEPS:
                reward = 100.0  # Survival bonus
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if movement == 1: self.player_rect.y -= self.PLAYER_SPEED # Up
        if movement == 2: self.player_rect.y += self.PLAYER_SPEED # Down
        if movement == 3: self.player_facing_right = False # Left
        if movement == 4: self.player_facing_right = True # Right

        self.player_rect.y = np.clip(self.player_rect.y, 0, self.GROUND_LEVEL - self.player_rect.height)

        if space_held and not self.prev_space_held and self.player_ammo > 0 and self.shoot_cooldown_timer <= 0:
            self._shoot()
        self.prev_space_held = space_held

    def _update_timers(self):
        if self.shoot_cooldown_timer > 0: self.shoot_cooldown_timer -= 1
        if self.player_hit_timer > 0: self.player_hit_timer -= 1
        self.zombie_spawn_timer -= 1
        self.ammo_spawn_timer -= 1

    def _shoot(self):
        # sound: laser_shoot.wav
        self.player_ammo -= 1
        self.shoot_cooldown_timer = self.PLAYER_SHOOT_COOLDOWN
        
        if self.player_facing_right:
            pos = self.player_rect.midright
            vel = [self.PROJECTILE_SPEED, 0]
        else:
            pos = self.player_rect.midleft
            vel = [-self.PROJECTILE_SPEED, 0]

        self.projectiles.append({'rect': pygame.Rect(pos[0], pos[1] - 2, 10, 4), 'vel': vel})
        self._create_particles(pos, 10, self.COLOR_MUZZLE_FLASH, 5, 4, vel[0] / 2)

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p['rect'].x += p['vel'][0]
            if not self.screen.get_rect().colliderect(p['rect']):
                self.projectiles.remove(p)

    def _update_zombies(self):
        kill_reward = 0
        damage_taken = 0
        zombie_speed = 0.5 + 0.1 * (self.steps // 1000)

        for z in self.zombies[:]:
            z['rect'].x -= zombie_speed
            
            if z['rect'].colliderect(self.player_rect):
                damage_taken += self.ZOMBIE_DAMAGE
                self._create_particles(z['rect'].center, 20, self.COLOR_ZOMBIE, 20, 3)
                self.zombies.remove(z)
                continue
            
            hit = False
            for p in self.projectiles[:]:
                if z['rect'].colliderect(p['rect']):
                    self.score += 10
                    kill_reward += 1.0
                    self._create_particles(z['rect'].center, 20, self.COLOR_ZOMBIE, 20, 3)
                    self.zombies.remove(z)
                    self.projectiles.remove(p)
                    # sound: zombie_die.wav
                    hit = True
                    break
            if hit: continue
            
            if z['rect'].right < 0: self.zombies.remove(z)
        return kill_reward, damage_taken

    def _update_ammo_crates(self):
        pickup_reward = 0
        for crate in self.ammo_crates[:]:
            if crate.colliderect(self.player_rect):
                self.player_ammo = min(self.PLAYER_MAX_AMMO, self.player_ammo + self.AMMO_REFILL)
                pickup_reward += 0.5
                self._create_particles(crate.center, 15, self.COLOR_AMMO, 25, 3, 0, True)
                self.ammo_crates.remove(crate)
                # sound: ammo_pickup.wav
        return pickup_reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0: self.particles.remove(p)

    def _spawn_zombies(self):
        spawn_interval = max(5, 20 - self.steps // 500)
        if self.zombie_spawn_timer <= 0:
            y_pos = self.GROUND_LEVEL - 40
            self.zombies.append({'rect': pygame.Rect(self.WIDTH, y_pos, 20, 40)})
            self.zombie_spawn_timer = self.np_random.integers(spawn_interval, spawn_interval + 10)

    def _spawn_ammo_crates(self):
        if self.ammo_spawn_timer <= 0:
            x = self.np_random.integers(200, self.WIDTH - 50)
            y = self.np_random.integers(100, self.GROUND_LEVEL - 40)
            crate_rect = pygame.Rect(x, y, 20, 20)
            if not any(c.colliderect(crate_rect.inflate(100, 100)) for c in self.ammo_crates):
                 self.ammo_crates.append(crate_rect)
            self.ammo_spawn_timer = self.np_random.integers(150, 300)

    def _generate_background(self):
        self.city_scape.clear()
        for _ in range(30):
            x = self.np_random.integers(0, self.WIDTH)
            w = self.np_random.integers(20, 80)
            h = self.np_random.integers(50, 200)
            y = self.GROUND_LEVEL - h
            self.city_scape.append(pygame.Rect(x, y, w, h))

    def _create_particles(self, pos, count, color, max_life, speed, x_bias=0, circular=False):
        for _ in range(count):
            if circular:
                angle = self.np_random.random() * 2 * math.pi
                vel = [math.cos(angle) * speed * self.np_random.random(), math.sin(angle) * speed * self.np_random.random()]
            else:
                vel = [self.np_random.random() * speed - speed/2 + x_bias, self.np_random.random() * speed - speed/2]
            
            self.particles.append({
                'pos': list(pos), 'vel': vel, 'life': self.np_random.integers(max_life/2, max_life), 'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for rect in self.city_scape: pygame.draw.rect(self.screen, self.COLOR_CITY, rect)
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_LEVEL, self.WIDTH, self.HEIGHT - self.GROUND_LEVEL))

        for crate in self.ammo_crates:
            pygame.draw.rect(self.screen, (0,0,0), crate.inflate(4, 4))
            pygame.draw.rect(self.screen, self.COLOR_AMMO, crate)
            pygame.draw.rect(self.screen, (255,255,255), crate, 1)

        for z in self.zombies:
            bob = math.sin(self.steps * 0.2 + z['rect'].x * 0.1) * 2
            render_rect = z['rect'].copy()
            render_rect.y += bob
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, render_rect)
            eye_x = render_rect.x + (5 if z['rect'].x > self.player_rect.x else 12)
            pygame.draw.rect(self.screen, (255,255,255), (eye_x, render_rect.y + 8, 3, 3))
        
        player_color = (255, 255, 255) if self.player_hit_timer > 0 and self.steps % 2 == 0 else self.COLOR_PLAYER
        bob = math.sin(self.steps * 0.3) * 2
        render_rect = self.player_rect.copy()
        render_rect.y += bob
        pygame.draw.rect(self.screen, player_color, render_rect)
        gun_rect = pygame.Rect(0, 0, 15, 6)
        gun_rect.centery = render_rect.centery - 5
        if self.player_facing_right: gun_rect.left = render_rect.centerx
        else: gun_rect.right = render_rect.centerx
        pygame.draw.rect(self.screen, player_color, gun_rect)

        for p in self.projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE, p['rect'])
            
        for p in self.particles:
            size = max(1, p['life'] / 4)
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(size))

    def _render_ui(self):
        health_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        bar_width, bar_height = 150, 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, bar_width * health_ratio, bar_height))

        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        time_left = max(0, (self.MAX_STEPS - self.steps) / 30.0) # Assume 30 FPS
        time_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH / 2 - time_text.get_width() / 2, 10))
        
        ammo_text = self.font_ui.render(f"AMMO: {self.player_ammo}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ammo_text, (10, self.HEIGHT - ammo_text.get_height() - 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            result_str = "SURVIVED!" if self.steps >= self.MAX_STEPS else "YOU DIED"
            result_text = self.font_title.render(result_str, True, (255, 50, 50))
            self.screen.blit(result_text, result_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2)))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "ammo": self.player_ammo
        }

    def validate_implementation(self):
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
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Human Player Controls ---
    # To map keyboard keys to the MultiDiscrete action space
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Zombie Survival")
    clock = pygame.time.Clock()

    while not done:
        # --- Create Action ---
        # Default action is no-op
        movement = 0
        space_held = 0
        shift_held = 0

        # Get keyboard state
        keys = pygame.key.get_pressed()
        for key, move_action in key_map.items():
            if keys[key]:
                movement = move_action
                break # Prioritize first key in map if multiple are pressed
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]

        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Render ---
        # The observation is already the rendered screen, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling & Timing ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Info: {info}")
    pygame.quit()