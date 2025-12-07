
# Generated: 2025-08-27T20:42:40.062635
# Source Brief: brief_02550.md
# Brief Index: 2550

        
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
        "Controls: ↑↓ to aim. Press space to fire. Hold shift to reload."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive waves of procedurally generated zombies in a side-view shooter using limited ammo and strategic reloads."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
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
        
        # Fonts
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GROUND = (50, 50, 60)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_ZOMBIE = (220, 50, 50)
        self.COLOR_BULLET = (255, 255, 0)
        self.COLOR_MUZZLE_FLASH = (255, 220, 150)
        self.COLOR_BLOOD = (180, 0, 0)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_HEALTH_BAR = (0, 200, 0)
        self.COLOR_HEALTH_BAR_BG = (100, 0, 0)
        self.COLOR_RELOAD = (255, 165, 0)

        # Game constants
        self.GROUND_LEVEL = self.HEIGHT - 50
        self.MAX_STEPS = 1000
        self.TOTAL_WAVES = 5
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.player_health = 0
        self.player_max_health = 100
        self.player_pos = (0, 0)
        self.player_aim_angle = 0.0
        self.player_max_ammo = 30
        self.player_ammo = 0
        self.shoot_cooldown = 0
        self.shoot_cooldown_max = 5 # 6 shots per second at 30fps
        self.reload_timer = 0
        self.reload_time_max = 60 # 2 seconds at 30fps
        self.current_wave = 0
        self.zombies = []
        self.projectiles = []
        self.particles = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.player_health = self.player_max_health
        self.player_pos = (70, self.GROUND_LEVEL)
        self.player_aim_angle = 0.0
        self.player_ammo = self.player_max_ammo
        self.shoot_cooldown = 0
        self.reload_timer = 0
        
        self.current_wave = 1
        self.zombies = []
        self.projectiles = []
        self.particles = []
        
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        
        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        shift_pressed = action[2] == 1
        
        # --- Update Timers ---
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        if self.reload_timer > 0:
            self.reload_timer -= 1
            if self.reload_timer == 0:
                self.player_ammo = self.player_max_ammo
                # sfx: reload_complete.wav
        
        # --- Handle Player Actions ---
        if not self.game_over:
            # Aiming
            if movement == 1:  # Up
                self.player_aim_angle -= 1.5
            elif movement == 2:  # Down
                self.player_aim_angle += 1.5
            self.player_aim_angle = max(-45, min(45, self.player_aim_angle))

            # Reloading
            if shift_pressed and self.player_ammo < self.player_max_ammo and self.reload_timer == 0:
                self.reload_timer = self.reload_time_max
                # sfx: reload_start.wav

            # Shooting
            if space_pressed and self.shoot_cooldown == 0 and self.player_ammo > 0 and self.reload_timer == 0:
                self.player_ammo -= 1
                self.shoot_cooldown = self.shoot_cooldown_max
                
                gun_tip = self._get_gun_tip()
                self.projectiles.append({'pos': list(gun_tip), 'angle': self.player_aim_angle})
                self._create_muzzle_flash(gun_tip)
                # sfx: shoot.wav
                if self.player_ammo == 0:
                    reward -= 0.01 # Small penalty for emptying clip
            elif space_pressed and self.player_ammo == 0 and self.reload_timer == 0 and self.shoot_cooldown == 0:
                # sfx: empty_clip_click.wav
                self.shoot_cooldown = self.shoot_cooldown_max # prevent spamming click sound
                reward -= 0.01

        # --- Update Game Logic ---
        reward += self._update_projectiles()
        reward += self._update_zombies()
        self._update_particles()
        
        # --- Wave Progression ---
        if not self.game_over and len(self.zombies) == 0:
            if self.current_wave == self.TOTAL_WAVES:
                self.win = True
                self.game_over = True
                reward += 100
            else:
                self.current_wave += 1
                self._spawn_wave()
                self.player_health = min(self.player_max_health, self.player_health + 10) # Small health bonus
        
        # --- Check Termination Conditions ---
        if self.player_health <= 0 and not self.game_over:
            self.game_over = True
            reward -= 100
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True
            reward -= 50 # Penalty for running out of time
            
        terminated = self.game_over
        
        # Update score based on reward
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _spawn_wave(self):
        num_zombies = 10 + (self.current_wave - 1) * 2
        zombie_speed = 0.5 + (self.current_wave - 1) * 0.05
        
        for _ in range(num_zombies):
            spawn_x = self.WIDTH + self.np_random.integers(50, 300)
            self.zombies.append({
                'pos': [spawn_x, self.GROUND_LEVEL],
                'health': 100,
                'speed': zombie_speed * self.np_random.uniform(0.8, 1.2),
                'anim_offset': self.np_random.uniform(0, 2 * math.pi)
            })

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            proj['pos'][0] += math.cos(math.radians(proj['angle'])) * 15
            proj['pos'][1] += math.sin(math.radians(proj['angle'])) * 15

            # Check for off-screen
            if not (0 <= proj['pos'][0] < self.WIDTH and 0 <= proj['pos'][1] < self.HEIGHT):
                self.projectiles.remove(proj)
                reward -= 0.01 # Miss penalty
                continue
                
            # Check for collision with zombies
            hit = False
            for zombie in self.zombies[:]:
                zombie_rect = pygame.Rect(zombie['pos'][0] - 10, zombie['pos'][1] - 40, 20, 40)
                if zombie_rect.collidepoint(proj['pos']):
                    zombie['health'] -= 50
                    self._create_blood_splatter(proj['pos'])
                    reward += 0.1 # Hit bonus
                    hit = True
                    if zombie['health'] <= 0:
                        self.zombies.remove(zombie)
                        reward += 1 # Kill bonus
                        # sfx: zombie_die.wav
                    else:
                        # sfx: bullet_hit.wav
                        pass
                    break # Bullet can only hit one zombie
            
            if hit:
                self.projectiles.remove(proj)
        return reward
        
    def _update_zombies(self):
        reward = 0
        for zombie in self.zombies:
            zombie['pos'][0] -= zombie['speed']
            # Simple bobbing animation for walking
            zombie['pos'][1] = self.GROUND_LEVEL + math.sin(self.steps * 0.2 + zombie['anim_offset']) * 2

            # Check collision with player
            player_rect = pygame.Rect(self.player_pos[0] - 10, self.player_pos[1] - 40, 20, 40)
            if player_rect.collidepoint(zombie['pos']):
                self.player_health -= 10
                self.zombies.remove(zombie)
                reward -= 1 # Damage penalty
                self._create_blood_splatter(self.player_pos, count=20)
                # sfx: player_hurt.wav
                if self.player_health <= 0:
                    self.player_health = 0
        return reward
        
    def _create_muzzle_flash(self, pos):
        # sfx: shoot.wav
        for _ in range(10):
            angle = self.np_random.uniform(0, 360)
            speed = self.np_random.uniform(2, 5)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(math.radians(angle)) * speed, math.sin(math.radians(angle)) * speed],
                'color': self.COLOR_MUZZLE_FLASH,
                'radius': self.np_random.uniform(4, 8),
                'life': 5
            })

    def _create_blood_splatter(self, pos, count=10):
        for _ in range(count):
            angle = self.np_random.uniform(-180, 0) # Upwards and backwards arc
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(math.radians(angle)) * speed, math.sin(math.radians(angle)) * speed],
                'color': self.COLOR_BLOOD,
                'radius': self.np_random.uniform(1, 4),
                'life': 20
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity for blood
            p['life'] -= 1
            p['radius'] *= 0.95
            if p['life'] <= 0 or p['radius'] < 0.5:
                self.particles.remove(p)

    def _get_gun_tip(self):
        gun_length = 30
        angle_rad = math.radians(self.player_aim_angle)
        gun_x = self.player_pos[0] + gun_length * math.cos(angle_rad)
        gun_y = self.player_pos[1] - 25 + gun_length * math.sin(angle_rad)
        return (gun_x, gun_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_LEVEL, self.WIDTH, self.HEIGHT - self.GROUND_LEVEL))

        # Draw zombies
        for zombie in self.zombies:
            pos_x, pos_y = int(zombie['pos'][0]), int(zombie['pos'][1])
            body_rect = pygame.Rect(pos_x - 7, pos_y - 35, 14, 35)
            head_pos = (pos_x, pos_y - 35)
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, body_rect)
            pygame.gfxdraw.filled_circle(self.screen, head_pos[0], head_pos[1], 8, self.COLOR_ZOMBIE)

        # Draw player
        p_x, p_y = int(self.player_pos[0]), int(self.player_pos[1])
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (p_x - 5, p_y - 30, 10, 30)) # Body
        pygame.gfxdraw.filled_circle(self.screen, p_x, p_y - 30, 7, self.COLOR_PLAYER) # Head
        
        # Draw gun
        gun_tip = self._get_gun_tip()
        gun_start = (p_x, p_y - 25)
        pygame.draw.line(self.screen, (100, 100, 100), gun_start, gun_tip, 6)

        # Draw projectiles
        for proj in self.projectiles:
            pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            pygame.draw.circle(self.screen, self.COLOR_BULLET, pos, 3)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 2, self.COLOR_BULLET)

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, p['color'])

    def _render_ui(self):
        # Health Bar
        health_ratio = self.player_health / self.player_max_health
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(200 * health_ratio), 20))
        health_text = self.font_small.render(f"HP: {self.player_health}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Ammo Counter
        ammo_text = self.font_small.render(f"AMMO: {self.player_ammo}/{self.player_max_ammo}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ammo_text, (10, 35))

        # Wave Counter
        wave_text = self.font_small.render(f"WAVE: {self.current_wave}/{self.TOTAL_WAVES}", True, self.COLOR_UI_TEXT)
        text_rect = wave_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(wave_text, text_rect)

        # Reload Indicator
        if self.reload_timer > 0:
            reload_text = self.font_small.render("RELOADING...", True, self.COLOR_RELOAD)
            reload_rect = reload_text.get_rect(center=(self.WIDTH // 2, 30))
            self.screen.blit(reload_text, reload_rect)
            
        # Game Over / Win Message
        if self.game_over:
            if self.win:
                msg = "YOU SURVIVED!"
                color = self.COLOR_PLAYER
            else:
                msg = "GAME OVER"
                color = self.COLOR_ZOMBIE
            
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "health": self.player_health,
            "ammo": self.player_ammo,
            "zombies_remaining": len(self.zombies),
        }
        
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Zombie Survival")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print("      ZOMBIE SURVIVAL      ")
    print("="*30)
    print(env.game_description)
    print("\n" + env.user_guide)
    print("="*30 + "\n")

    while running:
        # Get player input
        keys = pygame.key.get_pressed()
        
        movement = 0 # No-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        space_pressed = 1 if keys[pygame.K_SPACE] else 0
        shift_pressed = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_pressed, shift_pressed]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle window events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Check for game over
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            print("Resetting in 3 seconds...")
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        # Maintain 30 FPS
        clock.tick(30)
        
    pygame.quit()