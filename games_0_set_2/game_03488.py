
# Generated: 2025-08-27T23:31:36.694334
# Source Brief: brief_03488.md
# Brief Index: 3488

        
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
        "Controls: ←→ to turn, ↑ to accelerate, ↓ to brake. Hold Shift to drift."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced Tron-style racer. Dodge obstacles, drift through turns, and race against the clock."
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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)
        
        # Colors (Tron-inspired)
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_TRACK = (200, 255, 255)
        self.COLOR_PLAYER = (255, 50, 50)
        self.COLOR_PLAYER_GLOW = (255, 100, 100, 100)
        self.COLOR_OBSTACLE = (100, 100, 120)
        self.COLOR_CHECKPOINT = (255, 220, 0)
        self.COLOR_PARTICLE = (255, 255, 200)
        self.COLOR_UI = (255, 255, 255)

        # Game constants
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.TIME_LIMIT_SECONDS = 30
        self.MAX_HITS = 5
        self.CHECKPOINTS_TO_WIN = 5
        self.TRACK_WIDTH = 300
        
        # Player physics
        self.MIN_SPEED = 2.0
        self.MAX_SPEED = 12.0
        self.ACCELERATION = 0.3
        self.BRAKING = 0.5
        self.DRAG = 0.98
        self.TURN_SPEED = 0.05
        self.DRIFT_TURN_MULTIPLIER = 1.8
        self.DRIFT_FRICTION = 0.85
        self.NORMAL_FRICTION = 0.7
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel_x = None
        self.player_speed = None
        self.player_angle = None
        self.obstacles = None
        self.particles = None
        self.checkpoints = None
        self.steps = None
        self.score = None
        self.time_left = None
        self.obstacle_hits = None
        self.checkpoints_cleared = None
        self.obstacle_density = None
        self.game_over = None
        self.game_won = None
        self.last_hit_timer = None
        
        # Initialize state variables for the first time
        self.reset()

        # Validate implementation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = [self.WIDTH // 2, self.HEIGHT * 0.8]
        self.player_vel_x = 0.0
        self.player_speed = self.MIN_SPEED
        self.player_angle = 0.0
        
        self.obstacles = []
        self.particles = []
        self.checkpoints = []
        
        self.steps = 0
        self.score = 0
        self.time_left = self.TIME_LIMIT_SECONDS * self.FPS
        self.obstacle_hits = 0
        self.checkpoints_cleared = 0
        self.obstacle_density = 0.15 # Initial density
        
        self.game_over = False
        self.game_won = False
        self.last_hit_timer = 0

        self._generate_track_segment()
        self._generate_track_segment() # Generate two segments to start
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held = action[1] == 1 # Not used
        shift_held = action[2] == 1  # Boolean
        
        reward = 0.01 # Small survival reward
        
        self._update_player(movement, shift_held)
        self._update_world()
        
        event_reward = self._handle_events()
        reward += event_reward
        
        self.steps += 1
        self.time_left -= 1
        if self.last_hit_timer > 0:
            self.last_hit_timer -= 1
        
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.game_won:
                reward += 50
            elif self.obstacle_hits >= self.MAX_HITS:
                reward -= 100 # Larger penalty for hitting obstacles vs. time out
        
        self.score += reward
        
        if self.auto_advance:
            self.clock.tick(self.FPS)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_player(self, movement, shift_held):
        # --- Acceleration / Braking ---
        if movement == 1: # Up
            self.player_speed = min(self.MAX_SPEED, self.player_speed + self.ACCELERATION)
        elif movement == 2: # Down
            self.player_speed = max(self.MIN_SPEED, self.player_speed - self.BRAKING)
        
        # Apply drag
        self.player_speed *= self.DRAG
        self.player_speed = max(self.MIN_SPEED, self.player_speed)

        # --- Turning ---
        turn_multiplier = self.DRIFT_TURN_MULTIPLIER if shift_held else 1.0
        if movement == 3: # Left
            self.player_vel_x -= self.TURN_SPEED * turn_multiplier
        if movement == 4: # Right
            self.player_vel_x += self.TURN_SPEED * turn_multiplier
        
        # Apply friction
        friction = self.DRIFT_FRICTION if shift_held else self.NORMAL_FRICTION
        self.player_vel_x *= friction
        
        self.player_pos[0] += self.player_vel_x * self.player_speed
        
        # Keep player within track boundaries
        track_left = (self.WIDTH - self.TRACK_WIDTH) / 2
        track_right = (self.WIDTH + self.TRACK_WIDTH) / 2
        self.player_pos[0] = np.clip(self.player_pos[0], track_left + 15, track_right - 15)
        
        # Update visual angle for turning effect
        self.player_angle = np.clip(self.player_vel_x * -0.2, -0.5, 0.5)

        # Add drift particles
        if shift_held and abs(self.player_vel_x) > 0.5:
            # # Sound: Tire screech
            for _ in range(2):
                self._create_particle(
                    pos=[self.player_pos[0] + random.uniform(-5, 5), self.player_pos[1] + 10],
                    vel=[self.player_vel_x * 0.5, self.player_speed * 0.2],
                    life=random.randint(5, 10),
                    radius=random.uniform(1, 3),
                    color=(150, 150, 150)
                )

    def _update_world(self):
        # Move obstacles down
        for obs in self.obstacles:
            obs['y'] += self.player_speed
        self.obstacles = [obs for obs in self.obstacles if obs['y'] < self.HEIGHT + 50]
        
        # Move checkpoints down
        for cp in self.checkpoints:
            cp['y'] += self.player_speed
        
        # Move particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
        # Generate new segment if needed
        if not self.checkpoints or self.checkpoints[0]['y'] > -self.HEIGHT / 2:
            self._generate_track_segment()

    def _generate_track_segment(self):
        track_left = (self.WIDTH - self.TRACK_WIDTH) / 2
        num_obstacles = int(self.np_random.integers(3, 8) * (1 + self.obstacle_density))

        for _ in range(num_obstacles):
            self.obstacles.append({
                'x': track_left + self.np_random.random() * self.TRACK_WIDTH,
                'y': -self.np_random.random() * self.HEIGHT,
                'w': self.np_random.integers(40, 80),
                'h': 20
            })
        
        # Add a new checkpoint at the top of the new segment
        last_y = min([cp['y'] for cp in self.checkpoints]) if self.checkpoints else 0
        self.checkpoints.append({'y': last_y - self.HEIGHT, 'cleared': False})

    def _handle_events(self):
        reward = 0
        player_hitbox = pygame.Rect(self.player_pos[0] - 8, self.player_pos[1] - 12, 16, 24)
        
        # Checkpoints
        for cp in self.checkpoints:
            if not cp['cleared'] and player_hitbox.y < cp['y']:
                cp['cleared'] = True
                self.checkpoints_cleared += 1
                reward += 20
                self.obstacle_density += 0.05 # Increase difficulty
                # # Sound: Checkpoint cleared
                if self.checkpoints_cleared >= self.CHECKPOINTS_TO_WIN:
                    self.game_won = True
                break

        # Obstacles
        is_near_miss = False
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs['x'] - obs['w']/2, obs['y'] - obs['h']/2, obs['w'], obs['h'])
            if player_hitbox.colliderect(obs_rect):
                if self.last_hit_timer == 0:
                    self.obstacle_hits += 1
                    reward -= 10
                    self.last_hit_timer = self.FPS // 2 # 0.5s invulnerability
                    self.player_speed *= 0.5 # Slow down on hit
                    # # Sound: Crash/Explosion
                    for _ in range(30):
                        self._create_particle(
                            pos=list(player_hitbox.center),
                            vel=[random.uniform(-5, 5), random.uniform(-5, 5)],
                            life=random.randint(15, 30),
                            radius=random.uniform(1, 4),
                            color=self.COLOR_PLAYER
                        )
                break # only one hit per frame

            near_miss_rect = obs_rect.inflate(40, 40)
            if player_hitbox.colliderect(near_miss_rect):
                is_near_miss = True

        if is_near_miss and self.last_hit_timer == 0:
            reward += 5
            # # Sound: Whoosh
            for _ in range(3):
                self._create_particle(
                    pos=[self.player_pos[0], self.player_pos[1]],
                    vel=[random.uniform(-1, 1), random.uniform(-2, 0)],
                    life=random.randint(10, 20),
                    radius=random.uniform(1, 3),
                    color=self.COLOR_PARTICLE
                )
        
        return reward

    def _check_termination(self):
        return (
            self.obstacle_hits >= self.MAX_HITS or
            self.time_left <= 0 or
            self.steps >= self.MAX_STEPS or
            self.game_won
        )

    def _create_particle(self, pos, vel, life, radius, color):
        self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'radius': radius, 'color': color})

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Track boundaries
        track_left = (self.WIDTH - self.TRACK_WIDTH) / 2
        track_right = (self.WIDTH + self.TRACK_WIDTH) / 2
        pygame.draw.line(self.screen, self.COLOR_TRACK, (track_left, 0), (track_left, self.HEIGHT), 2)
        pygame.draw.line(self.screen, self.COLOR_TRACK, (track_right, 0), (track_right, self.HEIGHT), 2)

        # Checkpoints
        for cp in self.checkpoints:
            if 0 < cp['y'] < self.HEIGHT:
                pygame.gfxdraw.hline(self.screen, int(track_left), int(track_right), int(cp['y']), self.COLOR_CHECKPOINT)
                pygame.gfxdraw.hline(self.screen, int(track_left), int(track_right), int(cp['y'])+1, self.COLOR_CHECKPOINT)

        # Obstacles
        for obs in self.obstacles:
            if obs['y'] > -50:
                rect = pygame.Rect(obs['x'] - obs['w']/2, obs['y'] - obs['h']/2, obs['w'], obs['h'])
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20)) if p['life'] < 20 else 255
            color = p['color']
            if len(color) == 3: color = (*color, alpha)
            else: color = (color[0], color[1], color[2], alpha)
            
            surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(surf, (int(p['pos'][0] - p['radius']), int(p['pos'][1] - p['radius'])), special_flags=pygame.BLEND_RGBA_ADD)

        # Player
        if self.last_hit_timer == 0 or self.steps % 4 < 2: # Flicker when hit
            # Glow
            glow_surf = pygame.Surface((60, 60), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (30, 30), 20)
            self.screen.blit(glow_surf, (int(self.player_pos[0] - 30), int(self.player_pos[1] - 30)), special_flags=pygame.BLEND_RGBA_ADD)

            # Car body
            p1 = (0, -15)
            p2 = (-8, 10)
            p3 = (8, 10)
            
            # Rotate points
            angle = self.player_angle
            points = [p1, p2, p3]
            rotated_points = []
            for x, y in points:
                new_x = x * math.cos(angle) - y * math.sin(angle)
                new_y = x * math.sin(angle) + y * math.cos(angle)
                rotated_points.append((int(self.player_pos[0] + new_x), int(self.player_pos[1] + new_y)))
            
            pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Time
        time_text = f"TIME: {self.time_left / self.FPS:.1f}"
        time_surf = self.font_small.render(time_text, True, self.COLOR_UI)
        self.screen.blit(time_surf, (10, 10))

        # Hits
        hits_text = f"HITS: {self.obstacle_hits}/{self.MAX_HITS}"
        hits_surf = self.font_small.render(hits_text, True, self.COLOR_UI)
        self.screen.blit(hits_surf, (self.WIDTH - hits_surf.get_width() - 10, 10))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_UI)
        self.screen.blit(score_surf, (self.WIDTH / 2 - score_surf.get_width()/2, self.HEIGHT - 30))

        # Checkpoints
        cp_text = f"CHECKPOINTS: {self.checkpoints_cleared}/{self.CHECKPOINTS_TO_WIN}"
        cp_surf = self.font_small.render(cp_text, True, self.COLOR_UI)
        self.screen.blit(cp_surf, (10, 35))

        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
            color = self.COLOR_CHECKPOINT if self.game_won else self.COLOR_PLAYER
            msg_surf = self.font_large.render(msg, True, color)
            self.screen.blit(msg_surf, (self.WIDTH/2 - msg_surf.get_width()/2, self.HEIGHT/2 - msg_surf.get_height()/2))
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "obstacle_hits": self.obstacle_hits,
            "checkpoints_cleared": self.checkpoints_cleared,
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

if __name__ == "__main__":
    # To play the game manually, you need to map keyboard keys to the MultiDiscrete action space.
    # This is a simple example of how to do that.
    env = GameEnv()
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # Mapping from Pygame keys to action components
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Pygame setup for manual play
    pygame.display.set_caption("Arcade Racer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    while not done:
        # --- Action selection ---
        movement = 0 # Default no-op
        space_held = 0
        shift_held = 0
        
        # Get keyboard state
        keys = pygame.key.get_pressed()
        for key, move_action in key_map.items():
            if keys[key]:
                movement = move_action
                break # Prioritize first key in map (e.g., up over down)

        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]

        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already the rendered frame, so we just need to display it.
        # Pygame uses (width, height), numpy uses (height, width), so we need to transpose.
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    print(f"Game Over! Final Score: {total_reward:.2f}")
    env.close()