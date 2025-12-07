
# Generated: 2025-08-27T15:25:11.058481
# Source Brief: brief_00984.md
# Brief Index: 984

        
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
    """
    An isometric-2D zombie survival arena game.
    The player must survive waves of zombies by managing ammo and position.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: Arrow keys to move. Space to shoot. Shift to reload."
    )
    game_description = (
        "Survive waves of zombies in a dark arena. Manage your ammo, aim carefully, and don't get cornered!"
    )

    # Frame advance setting
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    # Colors
    COLOR_BG = (30, 30, 40)
    COLOR_ARENA = (50, 50, 60)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_OUTLINE = (200, 255, 255)
    COLOR_ZOMBIE = (80, 100, 40)
    COLOR_ZOMBIE_DAMAGED = (150, 50, 50)
    COLOR_PROJECTILE = (255, 255, 0)
    COLOR_BLOOD = (180, 0, 0)
    COLOR_MUZZLE_FLASH = (255, 220, 150)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HEALTH_BAR_BG = (80, 0, 0)
    COLOR_HEALTH_BAR_FG = (0, 200, 0)
    COLOR_RELOAD_BAR = (200, 150, 0)
    COLOR_SHADOW = (0, 0, 0, 90)
    # Game parameters
    MAX_STEPS = 2500
    TOTAL_WAVES = 5
    ARENA_PADDING = 20
    # Player
    PLAYER_RADIUS = 10
    PLAYER_SPEED = 3.0
    PLAYER_MAX_HEALTH = 100
    PLAYER_MAX_AMMO = 30
    RELOAD_TIME = 60  # in frames (2 seconds at 30fps)
    # Zombie
    ZOMBIE_RADIUS = 9
    ZOMBIE_BASE_SPEED = 0.8
    ZOMBIE_SPEED_WAVE_INCREMENT = 0.05
    ZOMBIE_BASE_COUNT = 10
    ZOMBIE_COUNT_WAVE_INCREMENT = 2
    ZOMBIE_HEALTH = 3
    ZOMBIE_DAMAGE = 10
    # Projectile
    PROJECTILE_RADIUS = 3
    PROJECTILE_SPEED = 10.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)

        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_ammo = None
        self.player_aim_angle = None
        self.player_reloading_timer = None
        self.zombies = None
        self.projectiles = None
        self.particles = None
        self.wave = None
        self.zombies_to_spawn_this_wave = None
        self.current_zombie_speed = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.last_space_held = None
        self.last_shift_held = None
        self.screen_shake = 0

        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.screen_shake = 0

        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_ammo = self.PLAYER_MAX_AMMO
        self.player_aim_angle = -math.pi / 2  # Start aiming up
        self.player_reloading_timer = 0

        self.zombies = []
        self.projectiles = []
        self.particles = []
        
        self.last_space_held = False
        self.last_shift_held = False

        self.wave = 1
        self._setup_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward += self._handle_input(movement, space_held, shift_held)

        # --- Update Game State ---
        self._update_player()
        reward += self._update_projectiles()
        reward += self._update_zombies()
        self._update_particles()
        
        if self.screen_shake > 0:
            self.screen_shake -= 1

        # --- Wave Progression ---
        if not self.zombies and self.zombies_to_spawn_this_wave == 0:
            if self.wave < self.TOTAL_WAVES:
                self.wave += 1
                self._setup_wave()
                reward += 100  # Wave cleared reward
                # sound: wave_complete.wav
            else:
                self.game_over = True # Victory

        # --- Termination ---
        self.steps += 1
        terminated = self._check_termination()
        if terminated and self.player_health <= 0:
            reward -= 100 # Death penalty

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _setup_wave(self):
        self.zombies_to_spawn_this_wave = self.ZOMBIE_BASE_COUNT + (self.wave - 1) * self.ZOMBIE_COUNT_WAVE_INCREMENT
        self.current_zombie_speed = self.ZOMBIE_BASE_SPEED + (self.wave - 1) * self.ZOMBIE_SPEED_WAVE_INCREMENT
        self._spawn_zombies(self.zombies_to_spawn_this_wave)

    def _handle_input(self, movement, space_held, shift_held):
        # --- Movement ---
        move_vec = np.array([0.0, 0.0])
        if movement in [1, 2, 3, 4]:
            if movement == 1: move_vec[1] -= 1
            elif movement == 2: move_vec[1] += 1
            elif movement == 3: move_vec[0] -= 1
            elif movement == 4: move_vec[0] += 1
            
            # Normalize for diagonal movement
            norm = np.linalg.norm(move_vec)
            if norm > 0:
                self.player_pos += (move_vec / norm) * self.PLAYER_SPEED
                self.player_aim_angle = math.atan2(move_vec[1], move_vec[0])
        
        self.player_pos[0] = np.clip(self.player_pos[0], self.ARENA_PADDING, self.SCREEN_WIDTH - self.ARENA_PADDING)
        self.player_pos[1] = np.clip(self.player_pos[1], self.ARENA_PADDING, self.SCREEN_HEIGHT - self.ARENA_PADDING)

        # --- Actions (on press) ---
        reward = 0
        shoot_pressed = space_held and not self.last_space_held
        reload_pressed = shift_held and not self.last_shift_held
        
        if shoot_pressed:
            if self.player_reloading_timer == 0:
                if self.player_ammo > 0:
                    self._fire_projectile()
                    # sound: player_shoot.wav
                else:
                    reward -= 0.01 # Penalty for trying to shoot with no ammo
                    # sound: empty_clip.wav
        
        if reload_pressed:
            if self.player_reloading_timer == 0 and self.player_ammo < self.PLAYER_MAX_AMMO:
                self.player_reloading_timer = self.RELOAD_TIME
                # sound: player_reload.wav

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        return reward
        
    def _fire_projectile(self):
        self.player_ammo -= 1
        start_pos = self.player_pos.copy()
        velocity = np.array([math.cos(self.player_aim_angle), math.sin(self.player_aim_angle)]) * self.PROJECTILE_SPEED
        self.projectiles.append({'pos': start_pos, 'vel': velocity})
        
        # Muzzle flash particle
        flash_pos = self.player_pos + np.array([math.cos(self.player_aim_angle), math.sin(self.player_aim_angle)]) * (self.PLAYER_RADIUS + 5)
        self.particles.append({'pos': flash_pos, 'type': 'muzzle_flash', 'life': 3})

    def _update_player(self):
        if self.player_reloading_timer > 0:
            self.player_reloading_timer -= 1
            if self.player_reloading_timer == 0:
                self.player_ammo = self.PLAYER_MAX_AMMO
                # sound: reload_complete.wav

    def _update_projectiles(self):
        reward = 0
        for p in self.projectiles[:]:
            p['pos'] += p['vel']
            
            # Wall collision
            if not (self.ARENA_PADDING < p['pos'][0] < self.SCREEN_WIDTH - self.ARENA_PADDING and \
                    self.ARENA_PADDING < p['pos'][1] < self.SCREEN_HEIGHT - self.ARENA_PADDING):
                self.projectiles.remove(p)
                reward -= 0.01 # Miss penalty
                continue
                
            # Zombie collision
            hit = False
            for z in self.zombies[:]:
                if np.linalg.norm(p['pos'] - z['pos']) < self.ZOMBIE_RADIUS + self.PROJECTILE_RADIUS:
                    z['health'] -= 1
                    z['hit_timer'] = 5 # Visual feedback for being hit
                    reward += 0.1 # Hit reward
                    self._create_blood_particles(p['pos'])
                    # sound: zombie_hit.wav
                    if z['health'] <= 0:
                        self.zombies.remove(z)
                        self.score += 10
                        reward += 1.0 # Kill reward
                        # sound: zombie_die.wav
                    
                    self.projectiles.remove(p)
                    hit = True
                    break
            if hit:
                continue
        return reward

    def _update_zombies(self):
        reward = 0
        for z in self.zombies:
            direction = self.player_pos - z['pos']
            norm = np.linalg.norm(direction)
            if norm > 1: # Avoid division by zero and jittering
                z['pos'] += (direction / norm) * self.current_zombie_speed
            
            if z['hit_timer'] > 0:
                z['hit_timer'] -= 1

            # Player collision
            if np.linalg.norm(self.player_pos - z['pos']) < self.PLAYER_RADIUS + self.ZOMBIE_RADIUS:
                if self.player_health > 0:
                    self.player_health -= self.ZOMBIE_DAMAGE
                    reward -= 1.0 # Damage penalty
                    self.screen_shake = 10
                    # sound: player_damage.wav
                self.player_health = max(0, self.player_health)

        return reward

    def _spawn_zombies(self, count):
        for _ in range(count):
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                x = self.np_random.uniform(0, self.SCREEN_WIDTH)
                y = -self.ZOMBIE_RADIUS
            elif edge == 1: # Bottom
                x = self.np_random.uniform(0, self.SCREEN_WIDTH)
                y = self.SCREEN_HEIGHT + self.ZOMBIE_RADIUS
            elif edge == 2: # Left
                x = -self.ZOMBIE_RADIUS
                y = self.np_random.uniform(0, self.SCREEN_HEIGHT)
            else: # Right
                x = self.SCREEN_WIDTH + self.ZOMBIE_RADIUS
                y = self.np_random.uniform(0, self.SCREEN_HEIGHT)

            self.zombies.append({'pos': np.array([x, y]), 'health': self.ZOMBIE_HEALTH, 'hit_timer': 0})
        self.zombies_to_spawn_this_wave = 0

    def _update_particles(self):
        for p in self.particles[:]:
            p['life'] -= 1
            if 'vel' in p:
                p['pos'] += p['vel']
            if p['life'] <= 0:
                self.particles.remove(p)
                
    def _create_blood_particles(self, pos):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'type': 'blood', 'life': self.np_random.integers(10, 20)})

    def _check_termination(self):
        if self.game_over: return True
        if self.player_health <= 0: return True
        if self.steps >= self.MAX_STEPS: return True
        if self.wave > self.TOTAL_WAVES: return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "health": self.player_health,
            "ammo": self.player_ammo,
            "zombies_left": len(self.zombies),
        }

    def _render_game(self):
        # --- Screen Shake ---
        render_offset = [0, 0]
        if self.screen_shake > 0:
            render_offset[0] = self.np_random.integers(-self.screen_shake, self.screen_shake + 1)
            render_offset[1] = self.np_random.integers(-self.screen_shake, self.screen_shake + 1)
        
        # --- Arena ---
        arena_rect = pygame.Rect(self.ARENA_PADDING, self.ARENA_PADDING, 
                                 self.SCREEN_WIDTH - 2 * self.ARENA_PADDING, 
                                 self.SCREEN_HEIGHT - 2 * self.ARENA_PADDING)
        pygame.draw.rect(self.screen, self.COLOR_ARENA, arena_rect.move(render_offset))

        # --- Shadows ---
        def draw_shadow(pos, radius):
            shadow_rect = pygame.Rect(pos[0] - radius, pos[1] - radius + 2, radius * 2, radius * 2)
            shadow_surf = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
            pygame.draw.ellipse(shadow_surf, self.COLOR_SHADOW, (0, 0, shadow_rect.width, shadow_rect.height))
            self.screen.blit(shadow_surf, shadow_rect.move(render_offset))

        for z in self.zombies:
            draw_shadow(z['pos'], self.ZOMBIE_RADIUS)
        draw_shadow(self.player_pos, self.PLAYER_RADIUS)

        # --- Game Elements ---
        for z in self.zombies:
            pos = (int(z['pos'][0] + render_offset[0]), int(z['pos'][1] + render_offset[1]))
            color = self.COLOR_ZOMBIE_DAMAGED if z['hit_timer'] > 0 else self.COLOR_ZOMBIE
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ZOMBIE_RADIUS, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ZOMBIE_RADIUS, color)

        for p in self.projectiles:
            pos = (int(p['pos'][0] + render_offset[0]), int(p['pos'][1] + render_offset[1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)

        # --- Player ---
        player_int_pos = (int(self.player_pos[0] + render_offset[0]), int(self.player_pos[1] + render_offset[1]))
        pygame.gfxdraw.filled_circle(self.screen, player_int_pos[0], player_int_pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_int_pos[0], player_int_pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER_OUTLINE)
        # Aiming line
        aim_end_pos = (
            player_int_pos[0] + int(math.cos(self.player_aim_angle) * self.PLAYER_RADIUS * 1.5),
            player_int_pos[1] + int(math.sin(self.player_aim_angle) * self.PLAYER_RADIUS * 1.5)
        )
        pygame.draw.line(self.screen, self.COLOR_PLAYER_OUTLINE, player_int_pos, aim_end_pos, 2)
        
        # --- Particles ---
        for p in self.particles:
            pos = (int(p['pos'][0] + render_offset[0]), int(p['pos'][1] + render_offset[1]))
            if p['type'] == 'muzzle_flash':
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], p['life'] * 3, self.COLOR_MUZZLE_FLASH)
            elif p['type'] == 'blood':
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], max(1, p['life'] // 4), self.COLOR_BLOOD)

    def _render_ui(self):
        # --- Health Bar ---
        health_pct = self.player_health / self.PLAYER_MAX_HEALTH
        bar_width = 150
        bar_height = 15
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (10, 10, int(bar_width * health_pct), bar_height))

        # --- Reload Bar ---
        if self.player_reloading_timer > 0:
            reload_pct = 1.0 - (self.player_reloading_timer / self.RELOAD_TIME)
            reload_bar_width = self.PLAYER_RADIUS * 2
            reload_bar_y = self.player_pos[1] - self.PLAYER_RADIUS - 10
            pygame.draw.rect(self.screen, self.COLOR_RELOAD_BAR, (self.player_pos[0] - self.PLAYER_RADIUS, reload_bar_y, int(reload_bar_width * reload_pct), 5))

        # --- Text Info ---
        ammo_text = self.font_small.render(f"AMMO: {self.player_ammo}/{self.PLAYER_MAX_AMMO}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ammo_text, (self.SCREEN_WIDTH - ammo_text.get_width() - 10, 10))
        
        wave_text = self.font_large.render(f"WAVE {self.wave}/{self.TOTAL_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH/2 - wave_text.get_width()/2, 10))

        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 30))

        # --- Game Over/Win Text ---
        if self._check_termination():
            if self.player_health <= 0:
                msg = "GAME OVER"
            elif self.wave > self.TOTAL_WAVES:
                msg = "YOU SURVIVED"
            else: # Timeout
                msg = "TIME'S UP"
            
            end_text = self.font_large.render(msg, True, (255, 50, 50))
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == self.observation_space.shape
        assert obs.dtype == self.observation_space.dtype
        assert isinstance(info, dict)
        
        # Test observation space after reset
        test_obs = self._get_observation()
        assert test_obs.shape == self.observation_space.shape
        assert test_obs.dtype == self.observation_space.dtype
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == self.observation_space.shape
        assert obs.dtype == self.observation_space.dtype
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # For interactive testing, we need a display
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "dummy" for no display
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a visible window for human play
    pygame.display.init()
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Zombie Survival")
    clock = pygame.time.Clock()

    done = False
    total_reward = 0
    
    print("\n" + "="*30)
    print("ZOMBIE SURVIVAL ARENA")
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # Human controls
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Handle window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()