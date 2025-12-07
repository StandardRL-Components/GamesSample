import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your ship. Hold Space to mine nearby asteroids."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a mining ship through asteroid fields. Extract valuable minerals while dodging collisions to get a high score."
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
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_ASTEROID = (120, 120, 120)
        self.COLOR_MINERAL = (255, 220, 0)
        self.COLOR_DEPOT = (0, 128, 255)
        self.COLOR_LASER = (255, 100, 100)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_EXPLOSION = [(255, 50, 50), (255, 150, 50), (255, 255, 255)]

        # Game constants
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 500
        self.INITIAL_LIVES = 3
        self.PLAYER_SPEED = 3.0
        self.PLAYER_DRAG = 0.90
        self.PLAYER_RADIUS = 12
        self.MINING_RANGE = 80
        self.INVULNERABILITY_DURATION = 90 # 3 seconds at 30fps

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_lives = 0
        self.player_invulnerable_timer = 0
        self.asteroids = []
        self.particles = []
        self.mineral_particles = []
        self.last_milestone_reward = 0
        self.asteroid_spawn_timer = 0
        self.asteroid_spawn_interval = 50
        self.max_asteroid_minerals = 10
        self.stars = []
        self.rng = np.random.default_rng()
        self.mining_beam_target_pos = None

        # self.reset() is called to set the initial state
        # self.validate_implementation() is a helper and not needed for the final version
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Do not create a new RNG if seed is None, to follow Gymnasium's recommendation
        # https://gymnasium.farama.org/api/env/#gymnasium.Env.reset

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 50)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_lives = self.INITIAL_LIVES
        self.player_invulnerable_timer = 0
        
        self.asteroids = []
        self.particles = []
        self.mineral_particles = []
        self.mining_beam_target_pos = None
        
        self.last_milestone_reward = 0
        self.asteroid_spawn_timer = 0
        self.asteroid_spawn_interval = 50
        self.max_asteroid_minerals = 10

        # Pre-generate stars for the background
        self.stars = [
            (
                self.rng.integers(0, self.WIDTH), 
                self.rng.integers(0, self.HEIGHT), 
                self.rng.integers(1, 3)
            ) for _ in range(100)
        ]
        
        # Initial asteroid spawn
        for _ in range(5):
            self._spawn_asteroid(random_pos=True)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        # --- 1. Handle Input ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1  # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1  # Right
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_vel += move_vec * 0.5
            if self.player_vel.length() > self.PLAYER_SPEED:
                self.player_vel.scale_to_length(self.PLAYER_SPEED)

        # --- 2. Update Game State ---
        self.steps += 1
        
        # Player movement
        self.player_vel *= self.PLAYER_DRAG
        self.player_pos += self.player_vel
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)

        # Invulnerability
        if self.player_invulnerable_timer > 0:
            self.player_invulnerable_timer -= 1

        # Asteroid spawning & progression
        self._update_progression()
        self.asteroid_spawn_timer += 1
        if self.asteroid_spawn_timer >= self.asteroid_spawn_interval:
            self.asteroid_spawn_timer = 0
            self._spawn_asteroid()

        # Mining
        self._handle_mining(space_held)

        # Update particles
        self._update_particles()
        
        # --- 3. Collisions & Rewards ---
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # Mining reward is handled in _handle_mining
        # We need to find the total minerals mined this frame
        mined_this_frame = sum(1 for p in self.mineral_particles if p['timer'] == p['duration'] -1)
        reward += mined_this_frame * 0.1

        # Milestone rewards
        current_milestone = self.score // 50
        if current_milestone > self.last_milestone_reward:
            milestone_reward = current_milestone * 10
            reward += milestone_reward
            self.last_milestone_reward = current_milestone
        
        # --- 4. Check Termination ---
        terminated = False
        truncated = False
        if self.player_lives <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
        elif self.score >= self.WIN_SCORE:
            self.score = self.WIN_SCORE
            reward += 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_progression(self):
        # Spawn rate increases slightly every 100 steps
        if self.steps > 0 and self.steps % 100 == 0:
            self.asteroid_spawn_interval = max(30, self.asteroid_spawn_interval - 2)
        
        # Max minerals increase every 250 steps
        if self.steps > 0 and self.steps % 250 == 0:
            self.max_asteroid_minerals = min(20, self.max_asteroid_minerals + 1)
    
    def _handle_mining(self, space_held):
        mining_target = None
        if space_held:
            closest_dist = float('inf')
            closest_asteroid = None
            for asteroid in self.asteroids:
                dist = self.player_pos.distance_to(asteroid['pos'])
                if dist < closest_dist:
                    closest_dist = dist
                    closest_asteroid = asteroid
            
            if closest_asteroid and closest_dist < self.MINING_RANGE + closest_asteroid['radius']:
                mining_target = closest_asteroid
        
        if mining_target:
            mining_target['minerals'] -= 1
            
            self.mineral_particles.append({
                'start_pos': mining_target['pos'].copy(),
                'end_pos': self.player_pos.copy(),
                'timer': 20,
                'duration': 20
            })
            
            if mining_target['minerals'] <= 0:
                self._create_explosion(mining_target['pos'], mining_target['radius'])
                self.asteroids.remove(mining_target)
        
        self.mining_beam_target_pos = mining_target['pos'] if mining_target else None

    def _handle_collisions(self):
        if self.player_invulnerable_timer > 0:
            return 0
        
        for asteroid in self.asteroids:
            dist = self.player_pos.distance_to(asteroid['pos'])
            if dist < self.PLAYER_RADIUS + asteroid['radius']:
                self.player_lives -= 1
                self.player_invulnerable_timer = self.INVULNERABILITY_DURATION
                self._create_explosion(self.player_pos, self.PLAYER_RADIUS * 1.5)
                return -1 # Collision penalty
        return 0

    def _spawn_asteroid(self, random_pos=False):
        minerals = self.rng.integers(5, self.max_asteroid_minerals + 1)
        radius = int(5 + minerals * 1.5)
        
        if random_pos:
            pos = pygame.Vector2(self.rng.integers(0, self.WIDTH), self.rng.integers(0, self.HEIGHT))
        else:
            edge = self.rng.integers(0, 4)
            if edge == 0: # Top
                pos = pygame.Vector2(self.rng.integers(0, self.WIDTH), -radius)
            elif edge == 1: # Bottom
                pos = pygame.Vector2(self.rng.integers(0, self.WIDTH), self.HEIGHT + radius)
            elif edge == 2: # Left
                pos = pygame.Vector2(-radius, self.rng.integers(0, self.HEIGHT))
            else: # Right
                pos = pygame.Vector2(self.WIDTH + radius, self.rng.integers(0, self.HEIGHT))
        
        num_points = self.rng.integers(7, 12)
        angles = sorted([self.rng.uniform(0, 2 * math.pi) for _ in range(num_points)])
        points = []
        for angle in angles:
            dist = self.rng.uniform(0.8, 1.2) * radius
            points.append(pygame.Vector2(dist, 0).rotate_rad(angle))

        self.asteroids.append({
            'pos': pos,
            'radius': radius,
            'minerals': minerals,
            'initial_minerals': minerals,
            'points': points
        })

    def _create_explosion(self, pos, size):
        num_particles = int(10 + size)
        for _ in range(num_particles):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 5)
            vel = pygame.Vector2(speed, 0).rotate_rad(angle)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'timer': self.rng.integers(15, 30),
                'color': self.COLOR_EXPLOSION[self.rng.integers(len(self.COLOR_EXPLOSION))]
            })

    def _update_particles(self):
        for p in self.mineral_particles[:]:
            p['timer'] -= 1
            if p['timer'] < 0:
                self.mineral_particles.remove(p)
                self.score += 1
            else:
                t = 1.0 - (p['timer'] / p['duration'])
                t = t * t # Ease in
                p['current_pos'] = p['start_pos'].lerp(self.player_pos, t)

        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Drag
            p['timer'] -= 1
            if p['timer'] < 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for x, y, size in self.stars:
            pygame.draw.rect(self.screen, (size * 40, size * 40, size * 50), (x, y, size, size))

        for asteroid in self.asteroids:
            points = [(p + asteroid['pos']) for p in asteroid['points']]
            if len(points) > 2:
                current_size_ratio = max(0.2, asteroid['minerals'] / asteroid['initial_minerals'])
                scaled_points = [( (p - asteroid['pos']) * current_size_ratio + asteroid['pos'] ) for p in points]
                pygame.gfxdraw.aapolygon(self.screen, [(int(p.x), int(p.y)) for p in scaled_points], self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, [(int(p.x), int(p.y)) for p in scaled_points], self.COLOR_ASTEROID)

        if self.mining_beam_target_pos:
            start_pos = (int(self.player_pos.x), int(self.player_pos.y))
            end_pos = (int(self.mining_beam_target_pos.x), int(self.mining_beam_target_pos.y))
            pygame.draw.aaline(self.screen, self.COLOR_LASER, start_pos, end_pos, 2)
            pygame.draw.aaline(self.screen, (255,255,255,50), start_pos, end_pos, 4)

        for p in self.mineral_particles:
            pos = (int(p['current_pos'].x), int(p['current_pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 2, self.COLOR_MINERAL)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 2, self.COLOR_MINERAL)

        is_invulnerable_flash = self.player_invulnerable_timer > 0 and (self.steps // 3) % 2 == 0
        if not is_invulnerable_flash:
            player_points = [
                pygame.Vector2(0, -self.PLAYER_RADIUS),
                pygame.Vector2(-self.PLAYER_RADIUS * 0.8, self.PLAYER_RADIUS * 0.8),
                pygame.Vector2(self.PLAYER_RADIUS * 0.8, self.PLAYER_RADIUS * 0.8),
            ]
            angle = self.player_vel.angle_to(pygame.Vector2(0, -1)) if self.player_vel.length() > 0.1 else 0
            rotated_points = [p.rotate(angle) + self.player_pos for p in player_points]
            int_points = [(int(p.x), int(p.y)) for p in rotated_points]
            pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_PLAYER)
            pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_PLAYER)

        for p in self.particles:
            size = int(p['timer'] / 5)
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), size)

    def _render_ui(self):
        score_text = self.font_small.render(f"MINERALS: {self.score}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        lives_text = self.font_small.render(f"LIVES: {self.player_lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))

        if self.game_over:
            if self.score >= self.WIN_SCORE:
                msg = "VICTORY!"
                color = self.COLOR_PLAYER
            else:
                msg = "GAME OVER"
                color = self.COLOR_EXPLOSION[0]
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            pygame.draw.rect(self.screen, (0,0,0,150), text_rect.inflate(20, 20))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # For manual play, you need a display.
    # On Linux/macOS: os.environ['SDL_VIDEODRIVER'] = 'x11'
    # On Windows: os.environ['SDL_VIDEODRIVER'] = 'windows'
    # If you are running this in a headless environment, the following will fail.
    try:
        os.environ['SDL_VIDEODRIVER'] = 'x11'
        env = GameEnv(render_mode="rgb_array")
        pygame.display.init()
        game_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Asteroid Miner")
    except pygame.error:
        print("Could not create Pygame display. Manual play is disabled.")
        print("To run headless, just instantiate the class: `env = GameEnv()`")
        game_screen = None

    if game_screen:
        obs, info = env.reset(seed=42)
        terminated = False
        truncated = False
        
        print("\n" + "="*30)
        print(f"GAME: {env.game_description}")
        print(f"CONTROLS: {env.user_guide}")
        print("="*30 + "\n")

        while not (terminated or truncated):
            movement = 0 # No-op
            space_held = 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space_held = 1

            action = [movement, space_held, 0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            game_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(30)

        print(f"Game Over! Final Score: {info['score']}")
        env.close()