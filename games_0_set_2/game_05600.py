
# Generated: 2025-08-28T05:31:01.360544
# Source Brief: brief_05600.md
# Brief Index: 5600

        
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
        "Controls: ↑/↓ to move. Press space to shoot. Hold shift to reload."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive waves of zombies in a frantic side-view arcade shooter. Manage your ammo and health to outlast the horde."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Colors ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GROUND = (60, 45, 40)
        self.COLOR_PLAYER = (70, 140, 200)
        self.COLOR_ZOMBIE = (90, 110, 80)
        self.COLOR_BULLET = (255, 255, 0)
        self.COLOR_MUZZLE_FLASH = (255, 200, 150)
        self.COLOR_HIT_SPARK = (255, 100, 100)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_HEALTH_FG = (50, 200, 50)
        self.COLOR_HEALTH_BG = (200, 50, 50)
        self.COLOR_RELOAD = (255, 165, 0)

        # --- Game Constants ---
        self.MAX_STEPS = 1000
        self.GROUND_HEIGHT = 50
        self.PLAYER_SPEED = 5
        self.PLAYER_WIDTH = 20
        self.PLAYER_HEIGHT = 40
        self.PLAYER_X = 80
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_MAX_AMMO = 20
        self.SHOT_COOLDOWN_FRAMES = 5 # 6 shots per second
        self.RELOAD_TIME_FRAMES = 60 # 2 seconds
        self.BULLET_SPEED = 15
        self.WAVE_CONFIG = {
            1: {'count': 5, 'speed': 1.0, 'spawn_delay': 60},
            2: {'count': 10, 'speed': 1.2, 'spawn_delay': 45},
            3: {'count': 15, 'speed': 1.5, 'spawn_delay': 30},
        }
        self.WAVE_TRANSITION_TIME = 90 # 3 seconds

        # Initialize state variables
        self.state_initialized = False
        self.reset()
        self.state_initialized = True
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if hasattr(self, 'np_random') and self.np_random is None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # --- Player State ---
        self.player_y = self.screen_height - self.GROUND_HEIGHT - self.PLAYER_HEIGHT
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_ammo = self.PLAYER_MAX_AMMO
        self.player_is_reloading = False
        self.player_reload_timer = 0
        self.shot_cooldown = 0
        
        # --- Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        
        # --- Wave State ---
        self.wave_number = 1
        self.zombies_to_spawn = self.WAVE_CONFIG[1]['count']
        self.zombies_spawned_this_wave = 0
        self.zombie_spawn_timer = 0
        self.wave_transition_timer = 0

        # --- Entity Lists ---
        self.zombies = []
        self.bullets = []
        self.particles = []

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if not self.state_initialized:
            self.reset()

        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.clock.tick(30) # Ensure 30 FPS for auto_advance

        # 1. Handle Input & Player Actions
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward += self._handle_player_actions(movement, space_pressed, shift_pressed)
        
        # 2. Update Game Logic
        self._update_timers()
        self._update_bullets()
        self._update_zombies()
        self._update_particles()
        
        # 3. Handle Spawning
        self._handle_wave_progression()

        # 4. Handle Collisions
        collision_rewards = self._handle_collisions()
        reward += collision_rewards

        # 5. Check Termination Conditions
        if self.player_health <= 0:
            self.game_over = True
            terminated = True
        elif self.win_condition_met:
            self.game_over = True
            terminated = True
            reward += 50.0 # Victory bonus
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_actions(self, movement, space_pressed, shift_pressed):
        # Movement
        if movement == 1: # Up
            self.player_y -= self.PLAYER_SPEED
        elif movement == 2: # Down
            self.player_y += self.PLAYER_SPEED
        self.player_y = np.clip(self.player_y, 0, self.screen_height - self.GROUND_HEIGHT - self.PLAYER_HEIGHT)

        # Shooting
        if space_pressed and self.player_ammo > 0 and self.shot_cooldown == 0 and not self.player_is_reloading:
            # Sfx: Gunshot
            self.player_ammo -= 1
            self.shot_cooldown = self.SHOT_COOLDOWN_FRAMES
            bullet_start_y = self.player_y + self.PLAYER_HEIGHT / 2
            self.bullets.append(pygame.Rect(self.PLAYER_X + self.PLAYER_WIDTH, bullet_start_y, 8, 4))
            # Muzzle flash
            self.particles.append({'type': 'flash', 'pos': (self.PLAYER_X + self.PLAYER_WIDTH, bullet_start_y), 'life': 2})
            return 0
        
        # Reloading
        if shift_pressed and not self.player_is_reloading and self.player_ammo < self.PLAYER_MAX_AMMO:
            # Sfx: Reload start
            self.player_is_reloading = True
            self.player_reload_timer = self.RELOAD_TIME_FRAMES
        
        return 0

    def _update_timers(self):
        if self.shot_cooldown > 0:
            self.shot_cooldown -= 1
        if self.player_is_reloading:
            self.player_reload_timer -= 1
            if self.player_reload_timer <= 0:
                # Sfx: Reload complete
                self.player_is_reloading = False
                self.player_ammo = self.PLAYER_MAX_AMMO
    
    def _update_bullets(self):
        for bullet in self.bullets:
            bullet.x += self.BULLET_SPEED
        
    def _update_zombies(self):
        wave_speed = self.WAVE_CONFIG[self.wave_number]['speed']
        for zombie in self.zombies:
            zombie['rect'].x -= wave_speed

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['life'] -= 1
            if p['type'] == 'spark':
                p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
                p['vel'] = (p['vel'][0] * 0.9, p['vel'][1] * 0.9 + 0.2) # Gravity
    
    def _handle_wave_progression(self):
        # Check if wave is complete
        if self.zombies_spawned_this_wave >= self.zombies_to_spawn and not self.zombies:
            if self.wave_transition_timer == 0: # Start transition
                if self.wave_number == 3:
                    self.win_condition_met = True
                    return
                self.score += 5.0 # Wave complete bonus
                self.wave_transition_timer = self.WAVE_TRANSITION_TIME
            else: # During transition
                self.wave_transition_timer -= 1
                if self.wave_transition_timer <= 0: # End transition, start next wave
                    self.wave_number += 1
                    config = self.WAVE_CONFIG[self.wave_number]
                    self.zombies_to_spawn = config['count']
                    self.zombies_spawned_this_wave = 0
                    self.zombie_spawn_timer = 0
        
        # Spawn zombies if wave is active
        elif self.zombies_spawned_this_wave < self.zombies_to_spawn and self.wave_transition_timer == 0:
            self.zombie_spawn_timer -= 1
            if self.zombie_spawn_timer <= 0:
                self.zombie_spawn_timer = self.WAVE_CONFIG[self.wave_number]['spawn_delay']
                self.zombies_spawned_this_wave += 1
                zombie_h = self.np_random.integers(30, 61)
                zombie_w = int(zombie_h * 0.6)
                zombie_y = self.screen_height - self.GROUND_HEIGHT - zombie_h
                self.zombies.append({
                    'rect': pygame.Rect(self.screen_width, zombie_y, zombie_w, zombie_h),
                    'health': 100
                })

    def _handle_collisions(self):
        reward = 0
        
        # Bullets vs Zombies
        bullets_to_remove = []
        zombies_to_remove = []
        for i, bullet in enumerate(self.bullets):
            for j, zombie in enumerate(self.zombies):
                if bullet.colliderect(zombie['rect']):
                    if i not in bullets_to_remove:
                        bullets_to_remove.append(i)
                    zombie['health'] -= 50
                    reward += 0.1 # Hit reward
                    # Sfx: Bullet hit
                    # Hit sparks
                    for _ in range(5):
                        self.particles.append({
                            'type': 'spark',
                            'pos': (bullet.centerx, bullet.centery),
                            'vel': (self.np_random.uniform(-3, 0), self.np_random.uniform(-2, 2)),
                            'life': self.np_random.integers(5, 15)
                        })

                    if zombie['health'] <= 0 and j not in zombies_to_remove:
                        zombies_to_remove.append(j)
                        reward += 1.0 # Kill reward
                        self.score += 1

        # Remove dead zombies
        if zombies_to_remove:
            self.zombies = [z for i, z in enumerate(self.zombies) if i not in zombies_to_remove]
        
        # Zombies vs Player
        player_rect = pygame.Rect(self.PLAYER_X, self.player_y, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        zombies_collided_player = []
        for i, zombie in enumerate(self.zombies):
            if zombie['rect'].colliderect(player_rect):
                self.player_health -= 25
                # Sfx: Player hurt
                zombies_collided_player.append(i)
            elif zombie['rect'].right < 0: # Zombie walked past
                zombies_collided_player.append(i)
        
        if zombies_collided_player:
            self.zombies = [z for i, z in enumerate(self.zombies) if i not in zombies_collided_player]

        # Wasted bullets
        new_bullets = []
        for i, bullet in enumerate(self.bullets):
            is_hit = i in bullets_to_remove
            if bullet.left > self.screen_width and not is_hit:
                reward -= 0.01 # Wasted shot penalty
            elif not is_hit:
                new_bullets.append(bullet)
        self.bullets = new_bullets
        
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.screen_height - self.GROUND_HEIGHT, self.screen_width, self.GROUND_HEIGHT))

        # Zombies
        for z in self.zombies:
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, z['rect'])
        
        # Player
        player_rect = pygame.Rect(self.PLAYER_X, int(self.player_y), self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        # Gun
        gun_rect = pygame.Rect(self.PLAYER_X + self.PLAYER_WIDTH, int(self.player_y) + self.PLAYER_HEIGHT/2 - 2, 10, 4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, gun_rect)

        # Bullets
        for b in self.bullets:
            pygame.draw.rect(self.screen, self.COLOR_BULLET, b)
        
        # Particles
        for p in self.particles:
            if p['type'] == 'flash':
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 8, self.COLOR_MUZZLE_FLASH)
            elif p['type'] == 'spark':
                pygame.draw.circle(self.screen, self.COLOR_HIT_SPARK, (int(p['pos'][0]), int(p['pos'][1])), 2)

    def _render_ui(self):
        # Health Bar
        health_ratio = np.clip(self.player_health / self.PLAYER_MAX_HEALTH, 0, 1)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (10, 10, int(200 * health_ratio), 20))
        
        # Ammo Count
        ammo_text_str = f"AMMO: {self.player_ammo}/{self.PLAYER_MAX_AMMO}"
        if self.player_is_reloading:
            ammo_text_str = "RELOADING..."
            # Reloading bar
            reload_ratio = (self.RELOAD_TIME_FRAMES - self.player_reload_timer) / self.RELOAD_TIME_FRAMES
            pygame.draw.rect(self.screen, self.COLOR_RELOAD, (10, 35, int(200 * reload_ratio), 5))

        ammo_text = self.font_small.render(ammo_text_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(ammo_text, (10, 35))

        # Wave Counter
        wave_text_str = f"WAVE: {self.wave_number}/3"
        if self.wave_transition_timer > 0:
            wave_text_str = f"WAVE {self.wave_number+1} IN {math.ceil(self.wave_transition_timer/30)}..."
        wave_text = self.font_small.render(wave_text_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.screen_width - wave_text.get_width() - 10, 10))
        
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.screen_width - score_text.get_width() - 10, 35))

        # Game Over / Win Message
        if self.game_over:
            msg_str = "YOU WIN!" if self.win_condition_met else "GAME OVER"
            color = self.COLOR_HEALTH_FG if self.win_condition_met else self.COLOR_HEALTH_BG
            msg_surf = self.font_large.render(msg_str, True, color)
            msg_rect = msg_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "health": self.player_health,
            "ammo": self.player_ammo
        }

    def close(self):
        pygame.quit()

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
    
    # Setup a window to display the game
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Zombie Wave Shooter")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(env.game_description)
    print(env.user_guide)

    while running:
        # --- Human Input ---
        movement = 0 # none
        space = 0 # released
        shift = 0 # released

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just need to display it
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
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}. Score: {info['score']}. Press 'R' to restart.")
            # Wait for reset
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        print("Resetting environment.")
                        obs, info = env.reset()
                        total_reward = 0
                        wait_for_reset = False
                clock.tick(30)
                
    env.close()