
# Generated: 2025-08-28T05:17:57.006409
# Source Brief: brief_05530.md
# Brief Index: 5530

        
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
        "Controls: Arrow keys to move. Hold Space to shoot. Press Shift to reload."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive for 60 seconds against an ever-growing horde of zombies in this top-down arena shooter."
    )

    # Frames auto-advance for smooth, real-time gameplay.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        
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
        self.font_large = pygame.font.Font(None, 50)
        
        # Colors
        self.COLOR_BG = (25, 25, 40)
        self.COLOR_WALL = (60, 60, 80)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_ZOMBIE = (220, 50, 50)
        self.COLOR_BULLET = (255, 255, 100)
        self.COLOR_BLOOD = (150, 0, 0)
        self.COLOR_MUZZLE_FLASH = (255, 230, 150)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_HEALTH_BAR_BG = (80, 0, 0)
        self.COLOR_HEALTH_BAR = (0, 180, 0)
        self.COLOR_AMMO_BAR = (200, 200, 0)
        self.COLOR_AMMO_BAR_BG = (80, 80, 0)
        self.COLOR_RELOAD = (255, 100, 0)

        # Game constants
        self.MAX_STEPS = self.FPS * 60  # 60 seconds
        
        # Player settings
        self.PLAYER_SIZE = 10
        self.PLAYER_SPEED = 4
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_MAX_AMMO = 12
        self.PLAYER_SHOOT_COOLDOWN = 4 # frames
        self.PLAYER_RELOAD_TIME = self.FPS * 1.5 # 1.5 seconds

        # Zombie settings
        self.ZOMBIE_SIZE = 12
        self.ZOMBIE_SPEED = 1.2
        self.ZOMBIE_HEALTH = 30
        self.ZOMBIE_DAMAGE = 25
        self.ZOMBIE_KNOCKBACK = 15

        # Bullet settings
        self.BULLET_SIZE = 3
        self.BULLET_SPEED = 15
        self.BULLET_DAMAGE = 15

        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_angle = None
        self.player_last_move_vec = None
        self.player_ammo = None
        self.player_shoot_timer = None
        self.reloading = None
        self.reload_timer = None
        self.zombies = None
        self.bullets = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_won = None
        self.zombie_spawn_timer = None
        self.zombie_spawn_interval = None
        self.reward_this_step = None
        self.game_timer = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_angle = -math.pi / 2  # Start facing up
        self.player_last_move_vec = np.array([0, -1], dtype=np.float32)
        self.player_ammo = self.PLAYER_MAX_AMMO
        self.player_shoot_timer = 0
        self.reloading = False
        self.reload_timer = 0

        self.zombies = []
        self.bullets = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.game_timer = 0.0

        self.zombie_spawn_timer = 0
        self.zombie_spawn_interval = self.FPS * 2 # Initial: 1 zombie every 2 seconds

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.reward_this_step = 0
        
        if self.game_over or self.game_won:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # --- Update Game Timers ---
        self.steps += 1
        self.game_timer += 1 / self.FPS
        if self.player_shoot_timer > 0: self.player_shoot_timer -= 1
        
        # --- Handle Player Actions ---
        self._handle_movement(movement)
        self._handle_shooting(space_held)
        self._handle_reloading(shift_held)
        
        # --- Update Game Entities ---
        self._update_bullets()
        self._update_zombies()
        self._update_spawner()
        self._update_particles()
        
        # --- Collision Detection ---
        self._check_collisions()
        
        # --- Check Termination Conditions ---
        terminated = self._check_termination()
        
        # --- Calculate Rewards ---
        # Small penalty for being close to zombies
        for z in self.zombies:
            if np.linalg.norm(self.player_pos - z['pos']) < 100:
                self.reward_this_step -= 0.001

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_movement(self, movement_action):
        move_vec = np.array([0, 0], dtype=np.float32)
        if movement_action == 1: move_vec[1] -= 1  # Up
        elif movement_action == 2: move_vec[1] += 1  # Down
        elif movement_action == 3: move_vec[0] -= 1  # Left
        elif movement_action == 4: move_vec[0] += 1  # Right
        
        if np.any(move_vec):
            # Normalize for consistent speed
            norm = np.linalg.norm(move_vec)
            if norm > 0:
                move_vec /= norm
                self.player_pos += move_vec * self.PLAYER_SPEED
                self.player_last_move_vec = move_vec
                self.player_angle = math.atan2(move_vec[1], move_vec[0])
        
        # Clamp player position to stay within bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

    def _handle_shooting(self, space_held):
        if space_held and self.player_ammo > 0 and self.player_shoot_timer == 0 and not self.reloading:
            self.player_ammo -= 1
            self.player_shoot_timer = self.PLAYER_SHOOT_COOLDOWN
            
            # Create bullet
            bullet_pos = self.player_pos + self.player_last_move_vec * (self.PLAYER_SIZE + 5)
            self.bullets.append({'pos': bullet_pos.copy(), 'vel': self.player_last_move_vec.copy()})
            # sfx: player_shoot.wav
            
            # Muzzle flash particle
            flash_pos = self.player_pos + self.player_last_move_vec * (self.PLAYER_SIZE + 2)
            self.particles.append({
                'pos': flash_pos, 'type': 'flash', 'radius': 8, 'life': 3
            })

    def _handle_reloading(self, shift_held):
        if shift_held and not self.reloading and self.player_ammo < self.PLAYER_MAX_AMMO:
            self.reloading = True
            self.reload_timer = self.PLAYER_RELOAD_TIME
            # sfx: reload_start.wav

        if self.reloading:
            self.reload_timer -= 1
            if self.reload_timer <= 0:
                self.reloading = False
                self.player_ammo = self.PLAYER_MAX_AMMO
                # sfx: reload_complete.wav

    def _update_bullets(self):
        for b in self.bullets:
            b['pos'] += b['vel'] * self.BULLET_SPEED
        # Remove bullets that are off-screen
        self.bullets = [b for b in self.bullets if 0 < b['pos'][0] < self.WIDTH and 0 < b['pos'][1] < self.HEIGHT]

    def _update_zombies(self):
        for z in self.zombies:
            direction = self.player_pos - z['pos']
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction /= norm
                z['pos'] += direction * self.ZOMBIE_SPEED
    
    def _update_spawner(self):
        self.zombie_spawn_timer -= 1
        if self.zombie_spawn_timer <= 0:
            # Spawn zombie at a random edge
            side = random.randint(0, 3)
            if side == 0: pos = np.array([random.uniform(0, self.WIDTH), -self.ZOMBIE_SIZE]) # Top
            elif side == 1: pos = np.array([random.uniform(0, self.WIDTH), self.HEIGHT + self.ZOMBIE_SIZE]) # Bottom
            elif side == 2: pos = np.array([-self.ZOMBIE_SIZE, random.uniform(0, self.HEIGHT)]) # Left
            else: pos = np.array([self.WIDTH + self.ZOMBIE_SIZE, random.uniform(0, self.HEIGHT)]) # Right
            
            self.zombies.append({'pos': pos, 'health': self.ZOMBIE_HEALTH})
            
            # Decrease spawn interval over time for difficulty progression
            # Initial: 2s. Min: 0.2s. Reaches min after ~50s.
            self.zombie_spawn_interval = max(self.FPS * 0.2, self.FPS * 2.0 - (self.game_timer * 0.7))
            self.zombie_spawn_timer = self.zombie_spawn_interval

    def _update_particles(self):
        for p in self.particles:
            p['life'] -= 1
            if p['type'] == 'blood':
                p['pos'] += p['vel']
                p['vel'] *= 0.9 # friction
                p['radius'] *= 0.95
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _check_collisions(self):
        # Bullet-Zombie collisions
        zombies_to_remove = []
        bullets_to_remove = []
        for i, b in enumerate(self.bullets):
            for j, z in enumerate(self.zombies):
                if i in bullets_to_remove or j in zombies_to_remove: continue
                dist = np.linalg.norm(b['pos'] - z['pos'])
                if dist < self.ZOMBIE_SIZE:
                    z['health'] -= self.BULLET_DAMAGE
                    self.reward_this_step += 0.1 # Reward for hitting
                    bullets_to_remove.append(i)
                    # sfx: zombie_hit.wav

                    # Blood splatter
                    for _ in range(5):
                        angle = random.uniform(0, 2 * math.pi)
                        speed = random.uniform(1, 4)
                        vel = np.array([math.cos(angle), math.sin(angle)]) * speed
                        self.particles.append({
                            'pos': z['pos'].copy(), 'type': 'blood', 'vel': vel,
                            'radius': random.uniform(2, 4), 'life': random.randint(10, 20)
                        })

                    if z['health'] <= 0:
                        zombies_to_remove.append(j)
                        self.score += 1
                        self.reward_this_step += 1 # Reward for killing
                        # sfx: zombie_death.wav
                        # Death particle explosion
                        for _ in range(20):
                            angle = random.uniform(0, 2 * math.pi)
                            speed = random.uniform(2, 6)
                            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
                            self.particles.append({
                                'pos': z['pos'].copy(), 'type': 'blood', 'vel': vel,
                                'radius': random.uniform(2, 5), 'life': random.randint(15, 30)
                            })
                    break # Bullet can only hit one zombie

        # Player-Zombie collisions
        for z in self.zombies:
            dist = np.linalg.norm(self.player_pos - z['pos'])
            if dist < self.PLAYER_SIZE + self.ZOMBIE_SIZE / 2:
                self.player_health -= self.ZOMBIE_DAMAGE
                # sfx: player_hurt.wav
                # Knockback player
                direction = self.player_pos - z['pos']
                norm = np.linalg.norm(direction)
                if norm > 0:
                    self.player_pos += (direction / norm) * self.ZOMBIE_KNOCKBACK
                if self.player_health <= 0:
                    self.game_over = True
                    self.reward_this_step -= 10 # Penalty for dying
                    break
        
        # Filter out removed entities
        self.bullets = [b for i, b in enumerate(self.bullets) if i not in bullets_to_remove]
        self.zombies = [z for i, z in enumerate(self.zombies) if i not in zombies_to_remove]

    def _check_termination(self):
        if self.player_health <= 0:
            self.game_over = True
            return True
        if self.game_timer >= 60:
            self.game_won = True
            self.reward_this_step += 50 # Reward for winning
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw arena walls for reference
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.HEIGHT), 2)
        
        # Draw particles
        for p in self.particles:
            if p['type'] == 'blood':
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), self.COLOR_BLOOD)
            elif p['type'] == 'flash':
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), self.COLOR_MUZZLE_FLASH)
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']*1.5), (*self.COLOR_MUZZLE_FLASH, 100))

        # Draw bullets
        for b in self.bullets:
            pos = (int(b['pos'][0]), int(b['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BULLET_SIZE, self.COLOR_BULLET)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BULLET_SIZE, self.COLOR_BULLET)

        # Draw zombies
        for z in self.zombies:
            pos = (int(z['pos'][0]), int(z['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ZOMBIE_SIZE, self.COLOR_ZOMBIE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ZOMBIE_SIZE, self.COLOR_ZOMBIE)

        # Draw player
        player_x, player_y = int(self.player_pos[0]), int(self.player_pos[1])
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, player_x, player_y, self.PLAYER_SIZE + 5, (*self.COLOR_PLAYER, 50))
        # Player body
        pygame.gfxdraw.filled_circle(self.screen, player_x, player_y, self.PLAYER_SIZE, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_x, player_y, self.PLAYER_SIZE, self.COLOR_PLAYER)
        # Aiming indicator
        end_x = player_x + math.cos(self.player_angle) * (self.PLAYER_SIZE + 3)
        end_y = player_y + math.sin(self.player_angle) * (self.PLAYER_SIZE + 3)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, (player_x, player_y), (end_x, end_y), 3)

    def _render_ui(self):
        # Health bar
        health_bar_width = 150
        health_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, health_bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(health_bar_width * health_ratio), 20))
        health_text = self.font_small.render("HEALTH", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Timer
        time_left = max(0, 60 - self.game_timer)
        timer_text = self.font_large.render(f"{time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH / 2 - timer_text.get_width() / 2, 5))
        
        # Score
        score_text = self.font_large.render(str(self.score), True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH / 2 - score_text.get_width() / 2, self.HEIGHT - 50))

        # Ammo
        if self.reloading:
            reload_progress = (self.PLAYER_RELOAD_TIME - self.reload_timer) / self.PLAYER_RELOAD_TIME
            pygame.draw.rect(self.screen, self.COLOR_AMMO_BAR_BG, (self.WIDTH - 160, 10, 150, 20))
            pygame.draw.rect(self.screen, self.COLOR_RELOAD, (self.WIDTH - 160, 10, int(150 * reload_progress), 20))
            ammo_text_str = "RELOADING"
        else:
            ammo_ratio = self.player_ammo / self.PLAYER_MAX_AMMO
            pygame.draw.rect(self.screen, self.COLOR_AMMO_BAR_BG, (self.WIDTH - 160, 10, 150, 20))
            pygame.draw.rect(self.screen, self.COLOR_AMMO_BAR, (self.WIDTH - 160, 10, int(150 * ammo_ratio), 20))
            ammo_text_str = f"AMMO: {self.player_ammo}/{self.PLAYER_MAX_AMMO}"
        
        ammo_text = self.font_small.render(ammo_text_str, True, self.COLOR_TEXT)
        text_rect = ammo_text.get_rect(center=(self.WIDTH - 85, 20))
        self.screen.blit(ammo_text, text_rect)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((100, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg = self.font_large.render("GAME OVER", True, self.COLOR_TEXT)
            self.screen.blit(msg, (self.WIDTH/2 - msg.get_width()/2, self.HEIGHT/2 - msg.get_height()/2))
        elif self.game_won:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 100, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg = self.font_large.render("YOU SURVIVED!", True, self.COLOR_TEXT)
            self.screen.blit(msg, (self.WIDTH/2 - msg.get_width()/2, self.HEIGHT/2 - msg.get_height()/2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "ammo": self.player_ammo,
            "time_left": max(0, 60 - self.game_timer)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
    # This block allows you to play the game manually for testing
    # It requires pygame to be installed and a display to be available
    import os
    # Check if a display is available, otherwise use a dummy driver
    try:
        pygame.display.init()
        pygame.font.init()
    except pygame.error:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a real display window
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Zombie Survival")
        clock = pygame.time.Clock()
        running = True
    except pygame.error:
        print("Pygame display not available. Cannot run interactive test.")
        running = False

    while running:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Handle window closing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}, Time Left: {info['time_left']:.2f}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(env.FPS)

    env.close()