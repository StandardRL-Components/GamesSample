
# Generated: 2025-08-27T22:15:02.053058
# Source Brief: brief_03062.md
# Brief Index: 3062

        
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
        "Controls: Arrow keys to move. Hold Space to fire. Shift does nothing."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro top-down shooter. Navigate the arena, dodge and destroy waves of geometric enemies to survive and get the highest score."
    )

    # Frames auto-advance for smooth, real-time gameplay.
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAX_STEPS = 1000
    NUM_ENEMIES = 20

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_OUTLINE = (200, 255, 220)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_PROJECTILE = (255, 255, 0)
    COLOR_PARTICLE_ORANGE = (255, 150, 0)
    COLOR_PARTICLE_YELLOW = (255, 255, 100)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_HEALTH_FG = (50, 200, 50)
    COLOR_HEALTH_BG = (100, 20, 20)
    
    # Game Physics
    PLAYER_SPEED = 4.0
    PLAYER_SIZE = 12
    PLAYER_FIRE_COOLDOWN = 6 # frames
    PLAYER_MAX_HEALTH = 100
    
    PROJECTILE_SPEED = 8.0
    PROJECTILE_SIZE = 3
    
    ENEMY_SPEED = 1.5
    ENEMY_SIZE = 10
    ENEMY_COLLISION_DAMAGE = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_health = None
        self.player_facing_direction = None
        self.player_fire_cooldown_timer = 0
        self.was_space_held = False

        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Player state
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_facing_direction = np.array([0, -1], dtype=np.float32) # Start facing up
        self.player_fire_cooldown_timer = 0
        self.was_space_held = False
        
        # Game objects
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self._spawn_enemies()
        
        return self._get_observation(), self._get_info()

    def _spawn_enemies(self):
        for _ in range(self.NUM_ENEMIES):
            # Ensure enemies don't spawn on the player
            while True:
                pos = self.np_random.random(2) * [self.SCREEN_WIDTH, self.SCREEN_HEIGHT]
                if np.linalg.norm(pos - self.player_pos) > 100:
                    break
            
            pattern = self.np_random.choice(['horizontal', 'vertical', 'circle', 'random_walk'])
            enemy_state = {
                'pos': pos.astype(np.float32),
                'size': self.ENEMY_SIZE,
                'pattern': pattern,
                'direction': self.np_random.choice([-1, 1]),
                'center': pos.copy(),
                'angle': self.np_random.random() * 2 * math.pi,
                'walk_timer': 0
            }
            self.enemies.append(enemy_state)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        reward = -0.01  # Small penalty for each step to encourage efficiency
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        self._handle_player_input(movement, space_held)
        self._update_projectiles()
        self._update_enemies()
        
        reward += self._handle_collisions()
        
        self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if self.player_health <= 0:
                reward -= 100 # Lost
            elif not self.enemies:
                reward += 100 # Won
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_input(self, movement, space_held):
        # --- Movement ---
        move_vector = np.array([0, 0], dtype=np.float32)
        if movement == 1: # Up
            move_vector[1] -= 1
        elif movement == 2: # Down
            move_vector[1] += 1
        elif movement == 3: # Left
            move_vector[0] -= 1
        elif movement == 4: # Right
            move_vector[0] += 1
        
        if np.any(move_vector):
            # Normalize for consistent speed, update facing direction
            self.player_facing_direction = move_vector / np.linalg.norm(move_vector)
            self.player_pos += self.player_facing_direction * self.PLAYER_SPEED

        # Clamp player position to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.SCREEN_WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.SCREEN_HEIGHT - self.PLAYER_SIZE)

        # --- Firing ---
        if self.player_fire_cooldown_timer > 0:
            self.player_fire_cooldown_timer -= 1
        
        # Fire on the frame the button is pressed down
        if space_held and not self.was_space_held and self.player_fire_cooldown_timer <= 0:
            # Sound: Pew!
            projectile_start_pos = self.player_pos + self.player_facing_direction * (self.PLAYER_SIZE + 1)
            self.projectiles.append({
                'pos': projectile_start_pos,
                'vel': self.player_facing_direction * self.PROJECTILE_SPEED
            })
            self.player_fire_cooldown_timer = self.PLAYER_FIRE_COOLDOWN
        self.was_space_held = space_held

    def _update_projectiles(self):
        # Update and filter out-of-bounds projectiles
        new_projectiles = []
        for p in self.projectiles:
            p['pos'] += p['vel']
            if 0 < p['pos'][0] < self.SCREEN_WIDTH and 0 < p['pos'][1] < self.SCREEN_HEIGHT:
                new_projectiles.append(p)
        self.projectiles = new_projectiles

    def _update_enemies(self):
        for enemy in self.enemies:
            pattern = enemy['pattern']
            pos = enemy['pos']
            
            if pattern == 'horizontal':
                pos[0] += self.ENEMY_SPEED * enemy['direction']
                if pos[0] < self.ENEMY_SIZE or pos[0] > self.SCREEN_WIDTH - self.ENEMY_SIZE:
                    enemy['direction'] *= -1
            elif pattern == 'vertical':
                pos[1] += self.ENEMY_SPEED * enemy['direction']
                if pos[1] < self.ENEMY_SIZE or pos[1] > self.SCREEN_HEIGHT - self.ENEMY_SIZE:
                    enemy['direction'] *= -1
            elif pattern == 'circle':
                enemy['angle'] += 0.05
                pos[0] = enemy['center'][0] + math.cos(enemy['angle']) * 40
                pos[1] = enemy['center'][1] + math.sin(enemy['angle']) * 40
            elif pattern == 'random_walk':
                enemy['walk_timer'] -= 1
                if enemy['walk_timer'] <= 0:
                    angle = self.np_random.random() * 2 * math.pi
                    enemy['walk_dir'] = np.array([math.cos(angle), math.sin(angle)])
                    enemy['walk_timer'] = self.np_random.integers(15, 30)
                pos += enemy.get('walk_dir', np.array([0,0])) * (self.ENEMY_SPEED * 0.75)
                # Clamp to a box around its center
                pos[0] = np.clip(pos[0], enemy['center'][0] - 50, enemy['center'][0] + 50)
                pos[1] = np.clip(pos[1], enemy['center'][1] - 50, enemy['center'][1] + 50)

            # Clamp all enemy positions to screen bounds
            pos[0] = np.clip(pos[0], self.ENEMY_SIZE, self.SCREEN_WIDTH - self.ENEMY_SIZE)
            pos[1] = np.clip(pos[1], self.ENEMY_SIZE, self.SCREEN_HEIGHT - self.ENEMY_SIZE)

    def _handle_collisions(self):
        reward = 0.0

        # Projectiles vs Enemies
        projectiles_to_remove = set()
        enemies_to_remove = set()
        
        for i, p in enumerate(self.projectiles):
            for j, e in enumerate(self.enemies):
                if j in enemies_to_remove: continue
                dist = np.linalg.norm(p['pos'] - e['pos'])
                if dist < self.ENEMY_SIZE + self.PROJECTILE_SIZE:
                    projectiles_to_remove.add(i)
                    enemies_to_remove.add(j)
                    reward += 1.0  # Reward for destroying an enemy
                    self.score += 1
                    self._create_explosion(e['pos'])
                    # Sound: Explosion!
                    break # Projectile can only hit one enemy
        
        if enemies_to_remove:
            self.enemies = [e for i, e in enumerate(self.enemies) if i not in enemies_to_remove]
        if projectiles_to_remove:
            self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in projectiles_to_remove]

        # Player vs Enemies
        for enemy in self.enemies:
            dist = np.linalg.norm(self.player_pos - enemy['pos'])
            if dist < self.PLAYER_SIZE + self.ENEMY_SIZE:
                self.player_health -= self.ENEMY_COLLISION_DAMAGE
                self._create_explosion(self.player_pos, count=10, intensity=0.5)
                # Sound: Player Hit!
                # For simplicity, we don't remove the enemy here, just damage player
                # A real game might have a brief invulnerability period
        
        self.player_health = max(0, self.player_health)
        return reward

    def _create_explosion(self, position, count=30, intensity=1.0):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 4.0 * intensity + 1.0
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': position.copy(),
                'vel': vel,
                'lifespan': self.np_random.integers(15, 30),
                'radius': self.np_random.random() * 3 + 2,
                'color': self.COLOR_PARTICLE_ORANGE if self.np_random.random() > 0.3 else self.COLOR_PARTICLE_YELLOW
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['lifespan'] -= 1
            p['radius'] *= 0.97
            if p['lifespan'] > 0 and p['radius'] > 0.5:
                active_particles.append(p)
        self.particles = active_particles

    def _check_termination(self):
        return self.player_health <= 0 or not self.enemies or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Particles (rendered first)
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30.0))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color
            )

        # Enemies
        for enemy in self.enemies:
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, 
                             (int(enemy['pos'][0] - enemy['size']), int(enemy['pos'][1] - enemy['size']),
                              enemy['size'] * 2, enemy['size'] * 2))

        # Projectiles
        for p in self.projectiles:
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'][0]), int(p['pos'][1]), self.PROJECTILE_SIZE, self.COLOR_PROJECTILE
            )
            pygame.gfxdraw.aacircle(
                self.screen, int(p['pos'][0]), int(p['pos'][1]), self.PROJECTILE_SIZE, self.COLOR_PROJECTILE
            )
        
        # Player
        angle = math.atan2(self.player_facing_direction[1], self.player_facing_direction[0]) + math.pi / 2
        points = []
        for i in range(3):
            a = angle + i * 2 * math.pi / 3
            p_size = self.PLAYER_SIZE if i == 0 else self.PLAYER_SIZE * 0.7
            x = self.player_pos[0] + math.sin(a) * p_size
            y = self.player_pos[1] - math.cos(a) * p_size
            points.append((x, y))
        
        int_points = [(int(p[0]), int(p[1])) for p in points]
        pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_PLAYER_OUTLINE)
        pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Health Bar
        health_pct = self.player_health / self.PLAYER_MAX_HEALTH
        bar_width = 150
        bar_height = 20
        bar_x = self.SCREEN_WIDTH - bar_width - 10
        bar_y = 10
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (bar_x, bar_y, int(bar_width * health_pct), bar_height))

        # Game Over / Win message
        if self.game_over:
            if not self.enemies:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            over_surf = self.font_game_over.render(msg, True, color)
            over_rect = over_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(over_surf, over_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "enemies_remaining": len(self.enemies),
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
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        assert len(self.enemies) == self.NUM_ENEMIES
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to 'human' to see the game being played
    render_mode = "human" # or "rgb_array"

    if render_mode == "human":
        # For human playback, we need a display
        import os
        # Check if running in a headless environment
        if os.environ.get('SDL_VIDEODRIVER', '') == 'dummy':
            print("Cannot run in human mode in a headless environment. Exiting.")
            exit()
        
        env = GameEnv(render_mode="rgb_array") # The env itself is always headless
        pygame.display.set_caption("Arcade Shooter")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        clock = pygame.time.Clock()
        
        obs, info = env.reset()
        terminated = False
        
        print("\n" + GameEnv.game_description)
        print(GameEnv.user_guide)
        
        while not terminated:
            # Map pygame keys to the MultiDiscrete action space
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
            
            # Render the observation from the environment to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
            clock.tick(30) # Run at 30 FPS
            
        print(f"Game Over! Final Info: {info}")
        env.close()
    
    else: # rgb_array test
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        print(f"Initial state: {info}")
        
        terminated = False
        total_reward = 0
        for _ in range(500):
            action = env.action_space.sample() # Random agent
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated:
                break
        
        print(f"Random agent finished. Final info: {info}, Total Reward: {total_reward:.2f}")
        env.close()