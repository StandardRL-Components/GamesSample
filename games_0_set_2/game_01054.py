
# Generated: 2025-08-27T15:43:18.150558
# Source Brief: brief_01054.md
# Brief Index: 1054

        
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
        "Controls: ←→ to run. Hold Shift to jump. Press Space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a relentless zombie horde for 60 seconds by running, jumping, and shooting in a side-scrolling action game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 60 * self.FPS  # 60 seconds
        self.GROUND_Y = self.HEIGHT - 60

        # Player constants
        self.PLAYER_SIZE = (20, 40)
        self.PLAYER_SPEED = 6
        self.PLAYER_JUMP_POWER = -16
        self.GRAVITY = 0.8
        self.MAX_HEALTH = 5
        self.SHOOT_COOLDOWN = 10  # frames

        # Zombie constants
        self.ZOMBIE_SIZE = (20, 40)
        self.ZOMBIE_SPEED_MIN = 1.0
        self.ZOMBIE_SPEED_MAX = 2.5
        
        # Colors
        self.COLOR_BG = (15, 15, 40)
        self.COLOR_GROUND = (70, 50, 40)
        self.COLOR_PLAYER = (50, 200, 50)
        self.COLOR_ZOMBIE = (200, 50, 50)
        self.COLOR_PROJECTILE = (255, 255, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_HEART = (255, 0, 0)
        
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
        self.font_large = pygame.font.Font(None, 48)
        
        # Initialize state variables
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = np.array([100.0, self.GROUND_Y - self.PLAYER_SIZE[1]])
        self.player_vel = np.array([0.0, 0.0])
        self.player_on_ground = True
        self.player_health = self.MAX_HEALTH
        
        self.zombies = []
        self.projectiles = []
        self.particles = []
        
        self.camera_x = 0
        self.shoot_timer = 0
        
        # Initial spawn rate: 1 per 3 seconds
        self.zombie_spawn_prob = 1.0 / (self.FPS * 3)
        # Difficulty increase: 0.001 per second
        self.zombie_spawn_increase = 0.001 / self.FPS
        
        self.last_space_state = 0
        self.last_shift_state = 0
        
        self.muzzle_flash = 0

        self._generate_background()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _generate_background(self):
        self.stars = [
            (self.np_random.integers(0, self.WIDTH * 2), self.np_random.integers(0, self.HEIGHT // 2))
            for _ in range(100)
        ]
        self.buildings_far = self._generate_buildings(20, 80, 20, 40)
        self.buildings_mid = self._generate_buildings(30, 120, 30, 60)
        self.buildings_near = self._generate_buildings(40, 180, 40, 80)

    def _generate_buildings(self, min_h, max_h, min_w, max_w):
        buildings = []
        x = -self.WIDTH
        while x < self.WIDTH * 3:
            w = self.np_random.integers(min_w, max_w)
            h = self.np_random.integers(min_h, max_h)
            buildings.append(pygame.Rect(x, self.GROUND_Y - h, w, h))
            x += w + self.np_random.integers(10, 30)
        return buildings

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0

        self._handle_input(action)
        
        self._update_player()
        self._update_zombies()
        self._update_projectiles()
        self._update_particles()
        
        collisions = self._handle_collisions()
        
        # Calculate reward based on events
        reward += 0.1  # Survival reward per frame
        reward += collisions['zombies_killed'] * 10
        reward -= collisions['player_hits'] * 1.0
        
        self.score += collisions['zombies_killed'] * 10

        self._update_camera()
        
        self.steps += 1
        
        # Check termination conditions
        terminated = self.player_health <= 0 or self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True
            if self.player_health > 0: # Survived the full time
                reward += 100

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel[0] = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_vel[0] = self.PLAYER_SPEED
        else:
            self.player_vel[0] = 0

        # Jumping (on rising edge of shift)
        if shift_held and not self.last_shift_state and self.player_on_ground:
            self.player_vel[1] = self.PLAYER_JUMP_POWER
            self.player_on_ground = False
            # Sfx: Jump sound

        # Shooting (on rising edge of space, with cooldown)
        if space_held and not self.last_space_state and self.shoot_timer <= 0:
            proj_pos = self.player_pos + np.array([self.PLAYER_SIZE[0] / 2, self.PLAYER_SIZE[1] / 2])
            self.projectiles.append({'pos': proj_pos, 'vel': np.array([20.0, 0.0])})
            self.shoot_timer = self.SHOOT_COOLDOWN
            self.muzzle_flash = 3 # frames
            # Sfx: Laser shot
            
        self.last_space_state = space_held
        self.last_shift_state = shift_held
        
    def _update_player(self):
        if self.shoot_timer > 0: self.shoot_timer -= 1
        if self.muzzle_flash > 0: self.muzzle_flash -= 1
            
        self.player_vel[1] += self.GRAVITY
        self.player_pos += self.player_vel
        
        if self.player_pos[1] + self.PLAYER_SIZE[1] >= self.GROUND_Y:
            self.player_pos[1] = self.GROUND_Y - self.PLAYER_SIZE[1]
            self.player_vel[1] = 0
            self.player_on_ground = True
            
        if self.player_pos[0] < self.camera_x:
            self.player_pos[0] = self.camera_x

    def _update_zombies(self):
        self.zombie_spawn_prob += self.zombie_spawn_increase
        if self.np_random.random() < self.zombie_spawn_prob:
            spawn_x = self.camera_x + self.WIDTH + 50
            speed = self.np_random.uniform(self.ZOMBIE_SPEED_MIN, self.ZOMBIE_SPEED_MAX)
            self.zombies.append({
                'pos': np.array([spawn_x, self.GROUND_Y - self.ZOMBIE_SIZE[1]]),
                'speed': speed,
                'anim_offset': self.np_random.integers(0, 100)
            })

        for z in self.zombies: z['pos'][0] -= z['speed']
        self.zombies = [z for z in self.zombies if z['pos'][0] + self.ZOMBIE_SIZE[0] > self.camera_x]

    def _update_projectiles(self):
        for p in self.projectiles: p['pos'] += p['vel']
        self.projectiles = [p for p in self.projectiles if 0 < p['pos'][0] - self.camera_x < self.WIDTH]

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def _create_explosion(self, pos, color, num_particles=20, max_speed=5):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(), 'vel': vel,
                'life': self.np_random.integers(10, 20), 'color': color
            })

    def _handle_collisions(self):
        player_rect = pygame.Rect(self.player_pos, self.PLAYER_SIZE)
        zombies_killed, player_hits = 0, 0

        for p in self.projectiles[:]:
            proj_rect = pygame.Rect(p['pos'], (10, 4))
            for z in self.zombies[:]:
                if proj_rect.colliderect(pygame.Rect(z['pos'], self.ZOMBIE_SIZE)):
                    self._create_explosion(z['pos'] + np.array([self.ZOMBIE_SIZE[0]/2, self.ZOMBIE_SIZE[1]/2]), self.COLOR_ZOMBIE)
                    self.zombies.remove(z)
                    self.projectiles.remove(p)
                    zombies_killed += 1
                    # Sfx: Explosion
                    break 
        
        for z in self.zombies[:]:
            if player_rect.colliderect(pygame.Rect(z['pos'], self.ZOMBIE_SIZE)):
                self.zombies.remove(z)
                self.player_health -= 1
                player_hits += 1
                self._create_explosion(self.player_pos + np.array([self.PLAYER_SIZE[0]/2, self.PLAYER_SIZE[1]/2]), self.COLOR_PLAYER, 10, 3)
                # Sfx: Player hit
        
        return {'zombies_killed': zombies_killed, 'player_hits': player_hits}

    def _update_camera(self):
        target_camera_x = self.player_pos[0] - self.WIDTH * 0.2
        self.camera_x += (target_camera_x - self.camera_x) * 0.1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x, y in self.stars:
            px = int((x - self.camera_x * 0.1) % (self.WIDTH * 2) - self.WIDTH)
            pygame.gfxdraw.pixel(self.screen, px, y, (200, 200, 255))

        self._render_building_layer(self.buildings_far, 0.25, (40, 40, 80))
        self._render_building_layer(self.buildings_mid, 0.5, (60, 60, 100))
        self._render_building_layer(self.buildings_near, 0.75, (80, 80, 120))
        
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))

    def _render_building_layer(self, buildings, factor, color):
        for b in buildings:
            render_rect = b.copy()
            render_rect.x = int(b.x - self.camera_x * factor)
            pygame.draw.rect(self.screen, color, render_rect)

    def _render_game(self):
        for p in self.particles:
            pos = (int(p['pos'][0] - self.camera_x), int(p['pos'][1]))
            pygame.draw.circle(self.screen, p['color'], pos, max(0, int(p['life']/4)))

        for z in self.zombies:
            z_pos = (int(z['pos'][0] - self.camera_x), int(z['pos'][1]))
            z_rect = pygame.Rect(z_pos, self.ZOMBIE_SIZE)
            z_rect.y += math.sin((self.steps + z['anim_offset']) * 0.2) * 2
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, z_rect)

        player_rect = pygame.Rect(int(self.player_pos[0] - self.camera_x), int(self.player_pos[1]), self.PLAYER_SIZE[0], self.PLAYER_SIZE[1])
        if self.player_vel[0] != 0 and self.player_on_ground: player_rect.height -= int(abs(math.sin(self.steps * 0.5) * 4))
        if not self.player_on_ground: player_rect.width, player_rect.height = int(self.PLAYER_SIZE[0] * 0.8), int(self.PLAYER_SIZE[1] * 1.2)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        
        for p in self.projectiles:
            pos = (int(p['pos'][0] - self.camera_x), int(p['pos'][1]))
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE, pygame.Rect(pos, (10, 4)))

        if self.muzzle_flash > 0:
            flash_pos = (int(self.player_pos[0] - self.camera_x + self.PLAYER_SIZE[0]), int(self.player_pos[1] + self.PLAYER_SIZE[1] / 2))
            pygame.draw.circle(self.screen, (255, 255, 150), flash_pos, 10)

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_left = max(0, (self.MAX_STEPS - self.steps) // self.FPS)
        time_text = self.font_large.render(f"TIME: {time_left}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(centerx=self.WIDTH / 2, y=10)
        self.screen.blit(time_text, time_rect)

        for i in range(self.MAX_HEALTH):
            heart_pos = (self.WIDTH - 30 - i * 35, 15)
            is_filled = i < self.player_health
            color = self.COLOR_HEART if is_filled else (100, 100, 100)
            width = 0 if is_filled else 2
            pygame.draw.polygon(self.screen, color, [(heart_pos[0]+12, heart_pos[1]+5), (heart_pos[0], heart_pos[1]+15), (heart_pos[0]+12, heart_pos[1]+25), (heart_pos[0]+24, heart_pos[1]+15)], width)
            pygame.draw.circle(self.screen, color, (heart_pos[0]+6, heart_pos[1]+6), 6, width)
            pygame.draw.circle(self.screen, color, (heart_pos[0]+18, heart_pos[1]+6), 6, width)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
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

# Example usage to test the environment visually
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Zombie Horde Survival")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action = [movement, keys[pygame.K_SPACE], keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

    env.close()