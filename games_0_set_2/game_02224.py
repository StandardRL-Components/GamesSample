
# Generated: 2025-08-27T19:40:52.907075
# Source Brief: brief_02224.md
# Brief Index: 2224

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move. Press space to fire at the nearest zombie."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive waves of procedurally generated zombies in a top-down arena shooter for 5 minutes."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.PLAYER_SIZE = 20
        self.PLAYER_SPEED = 4
        self.PLAYER_HEALTH_MAX = 100
        self.ZOMBIE_SIZE = 20
        self.ZOMBIE_SPEED = 1.5
        self.ZOMBIE_HEALTH_MAX = 40
        self.ZOMBIE_DAMAGE = 10
        self.BULLET_SIZE = 4
        self.BULLET_SPEED = 10
        self.BULLET_DAMAGE = 20
        self.SHOOT_COOLDOWN = 5  # in steps
        self.MAX_ZOMBIES = 50
        self.MAX_STEPS = 3000
        self.ZOMBIE_SPAWN_INTERVAL = 20

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 50)
        self.COLOR_ZOMBIE = (255, 50, 50)
        self.COLOR_ZOMBIE_GLOW = (255, 50, 50, 50)
        self.COLOR_BULLET = (255, 255, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR = (0, 200, 80)
        self.COLOR_HEALTH_BAR_BG = (100, 0, 0)
        self.COLOR_HIT_FLASH = (255, 0, 0, 100)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font_small = pygame.font.SysFont('Consolas', 20, bold=True)
        self.font_large = pygame.font.SysFont('Consolas', 30, bold=True)
        
        # Initialize state variables
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = [self.WIDTH / 2, self.HEIGHT / 2]
        self.player_health = self.PLAYER_HEALTH_MAX
        self.player_rect = pygame.Rect(0,0,0,0) # Initialized properly below
        self.player_rect.size = (self.PLAYER_SIZE, self.PLAYER_SIZE)
        self.player_rect.center = self.player_pos
        
        self.zombies = []
        self.bullets = []
        self.particles = []
        
        self.shoot_cooldown_timer = 0
        self.prev_space_held = False
        self.hit_flash_timer = 0
        
        self.zombie_spawn_timer = self.ZOMBIE_SPAWN_INTERVAL
        self.zombies_per_spawn = 1
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.1  # Reward for surviving one step
        self.steps += 1

        self._update_timers()
        reward += self._handle_input(action)
        self._update_entities()
        self._handle_spawning()
        reward += self._handle_collisions()

        if self.steps > 0 and self.steps % 600 == 0:
            reward += 5.0

        terminated = False
        if self.player_health <= 0:
            reward = -100.0
            terminated = True
            self.game_over = True
            # SFX: Player death
        elif self.steps >= self.MAX_STEPS:
            reward += 100.0
            terminated = True
            self.game_over = True
            # SFX: Win fanfare

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_timers(self):
        if self.shoot_cooldown_timer > 0: self.shoot_cooldown_timer -= 1
        if self.hit_flash_timer > 0: self.hit_flash_timer -= 1
        self.zombie_spawn_timer -= 1

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward_penalty = 0
        
        move_vec = [0, 0]
        if movement == 1: move_vec[1] -= 1
        elif movement == 2: move_vec[1] += 1
        elif movement == 3: move_vec[0] -= 1
        elif movement == 4: move_vec[0] += 1
        
        self.player_pos[0] += move_vec[0] * self.PLAYER_SPEED
        self.player_pos[1] += move_vec[1] * self.PLAYER_SPEED

        if movement != 0 and self.zombies:
            nearest_zombie = self._find_nearest_zombie()
            if nearest_zombie:
                vec_to_zombie = self._get_vector(self.player_pos, nearest_zombie['pos'], normalize=False)
                dot_product = move_vec[0] * vec_to_zombie[0] + move_vec[1] * vec_to_zombie[1]
                if dot_product < 0: reward_penalty -= 0.2

        if space_held and not self.prev_space_held and self.shoot_cooldown_timer == 0:
            # SFX: Player shoot
            nearest_zombie = self._find_nearest_zombie()
            if nearest_zombie:
                self.shoot_cooldown_timer = self.SHOOT_COOLDOWN
                direction = self._get_vector(self.player_pos, nearest_zombie['pos'])
                self.bullets.append({
                    'pos': list(self.player_pos),
                    'vel': [direction[0] * self.BULLET_SPEED, direction[1] * self.BULLET_SPEED],
                    'rect': pygame.Rect(0,0,self.BULLET_SIZE, self.BULLET_SIZE)
                })
                self._spawn_particles(self.player_pos, self.COLOR_BULLET, count=1, life=2, size=12, speed=0)

        self.prev_space_held = space_held
        return reward_penalty

    def _update_entities(self):
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE/2, self.WIDTH - self.PLAYER_SIZE/2)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE/2, self.HEIGHT - self.PLAYER_SIZE/2)
        self.player_rect.center = self.player_pos

        for z in self.zombies:
            direction = self._get_vector(z['pos'], self.player_pos)
            wobble = self.np_random.uniform(-0.5, 0.5, 2)
            wobble_dir = np.array([direction[1], -direction[0]])
            
            z['pos'][0] += (direction[0] + wobble_dir[0] * wobble[0]) * self.ZOMBIE_SPEED
            z['pos'][1] += (direction[1] + wobble_dir[1] * wobble[1]) * self.ZOMBIE_SPEED
            z['rect'].center = z['pos']

        for b in self.bullets:
            b['pos'][0] += b['vel'][0]
            b['pos'][1] += b['vel'][1]
            b['rect'].center = b['pos']
            
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _handle_spawning(self):
        if self.steps > 0 and self.steps % 600 == 0:
            self.zombies_per_spawn += 1

        if self.zombie_spawn_timer <= 0 and len(self.zombies) < self.MAX_ZOMBIES:
            self.zombie_spawn_timer = self.ZOMBIE_SPAWN_INTERVAL
            for _ in range(self.zombies_per_spawn):
                if len(self.zombies) >= self.MAX_ZOMBIES: break
                
                edge = self.np_random.integers(4)
                if edge == 0: pos = [self.np_random.uniform(0, self.WIDTH), -self.ZOMBIE_SIZE]
                elif edge == 1: pos = [self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ZOMBIE_SIZE]
                elif edge == 2: pos = [-self.ZOMBIE_SIZE, self.np_random.uniform(0, self.HEIGHT)]
                else: pos = [self.WIDTH + self.ZOMBIE_SIZE, self.np_random.uniform(0, self.HEIGHT)]
                
                self.zombies.append({
                    'pos': pos, 'health': self.ZOMBIE_HEALTH_MAX,
                    'rect': pygame.Rect(0,0,self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
                })

    def _handle_collisions(self):
        reward = 0
        bullets_to_remove, zombies_to_remove = set(), set()

        for i, b in enumerate(self.bullets):
            for j, z in enumerate(self.zombies):
                if j in zombies_to_remove: continue
                if b['rect'].colliderect(z['rect']):
                    z['health'] -= self.BULLET_DAMAGE
                    self._spawn_particles(b['pos'], self.COLOR_BULLET, count=3, life=5, size=3)
                    bullets_to_remove.add(i)
                    if z['health'] <= 0:
                        reward += 1.0
                        self._spawn_particles(z['pos'], self.COLOR_ZOMBIE, count=15, life=15, size=4)
                        zombies_to_remove.add(j)
                    break
        
        for j, z in enumerate(self.zombies):
            if j in zombies_to_remove: continue
            if self.player_rect.colliderect(z['rect']):
                self.player_health = max(0, self.player_health - self.ZOMBIE_DAMAGE)
                self.hit_flash_timer = 4
                self._spawn_particles(self.player_pos, self.COLOR_ZOMBIE, count=10, life=10, size=5)
                zombies_to_remove.add(j)

        self.bullets = [b for i, b in enumerate(self.bullets) if i not in bullets_to_remove and 0 < b['pos'][0] < self.WIDTH and 0 < b['pos'][1] < self.HEIGHT]
        if zombies_to_remove:
            self.zombies = [z for i, z in enumerate(self.zombies) if i not in zombies_to_remove]
        self.particles = [p for p in self.particles if p['life'] > 0]
        
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for p in self.particles:
            size = max(0, p['size'] * (p['life'] / p['max_life']))
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(size))

        for z in self.zombies:
            pos = (int(z['pos'][0]), int(z['pos'][1]))
            size = int(self.ZOMBIE_SIZE * 0.7)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_ZOMBIE_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_ZOMBIE_GLOW)
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, z['rect'])

        for b in self.bullets:
            pygame.draw.circle(self.screen, self.COLOR_BULLET, (int(b['pos'][0]), int(b['pos'][1])), self.BULLET_SIZE)

        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        size = int(self.PLAYER_SIZE * 1.2)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_PLAYER_GLOW)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect)

    def _render_ui(self):
        if self.hit_flash_timer > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            flash_surface.fill(self.COLOR_HIT_FLASH)
            self.screen.blit(flash_surface, (0, 0))

        health_ratio = max(0, self.player_health / self.PLAYER_HEALTH_MAX)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(200 * health_ratio), 20))
        
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        time_left = max(0, (self.MAX_STEPS - self.steps) / 10.0)
        timer_text = self.font_large.render(f"{int(time_left // 60):02}:{int(time_left % 60):02}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.WIDTH/2 - timer_text.get_width()/2, 10))

        if self.game_over:
            end_text = "YOU SURVIVED!" if self.player_health > 0 else "GAME OVER"
            end_surf = self.font_large.render(end_text, True, self.COLOR_UI_TEXT)
            self.screen.blit(end_surf, (self.WIDTH/2 - end_surf.get_width()/2, self.HEIGHT/2 - end_surf.get_height()/2))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "health": self.player_health, "zombies": len(self.zombies)}

    def _find_nearest_zombie(self):
        if not self.zombies: return None
        player_pos_np = np.array(self.player_pos)
        zombie_positions = np.array([z['pos'] for z in self.zombies])
        distances = np.linalg.norm(zombie_positions - player_pos_np, axis=1)
        return self.zombies[np.argmin(distances)]

    def _get_vector(self, pos1, pos2, normalize=True):
        vec = np.array(pos2) - np.array(pos1)
        if normalize:
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else np.array([0, 0])
        return vec

    def _spawn_particles(self, pos, color, count, life, size, speed=2):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed * self.np_random.uniform(0.5, 1.5)
            self.particles.append({'pos': np.array(pos), 'vel': vel, 'life': life, 'max_life': life, 'color': color, 'size': size})
            
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,) and self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3) and test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3) and isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3) and isinstance(reward, (int, float))
        assert isinstance(term, bool) and trunc is False and isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    try:
        env = GameEnv()
        pygame.display.init()
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Zombie Survival")
        clock = pygame.time.Clock()
        obs, info = env.reset()
        done = False
        print("\n" + "="*30 + "\n" + "MANUAL PLAYBACK".center(30) + "\n" + "="*30)
        print(env.user_guide)

        while not done:
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            action = [movement, 1 if keys[pygame.K_SPACE] else 0, 1 if keys[pygame.K_LSHIFT] else 0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT: done = True
            clock.tick(10)
        
        print("Game Over!")
        print(f"Final Info: {info}")

    except Exception as e:
        print(f"\nCould not start manual playback (this is expected in a headless environment).")
        print(f"Error: {e}")
        print("The environment class is valid and ready for agent-based training.")
    
    env.close()