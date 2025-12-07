
# Generated: 2025-08-27T14:41:18.165121
# Source Brief: brief_00758.md
# Brief Index: 758

        
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
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold space to activate the mining beam."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a spaceship, mine asteroids for minerals, and reach the target yield of 50 before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.FPS = 50

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 60)
        self.COLOR_TIMER_NORMAL = (255, 255, 255)
        self.COLOR_TIMER_WARN = (255, 255, 0)
        self.COLOR_TIMER_CRITICAL = (255, 0, 0)
        self.COLOR_MINERAL = (255, 220, 0)
        self.COLOR_BEAM = (100, 200, 255, 150)
        self.ASTEROID_TYPES = [
            {'color': (100, 100, 100), 'minerals': 1, 'radius': 12, 'score_val': 1},
            {'color': (140, 120, 100), 'minerals': 2, 'radius': 16, 'score_val': 2},
            {'color': (200, 60, 60),   'minerals': 3, 'radius': 20, 'score_val': 3},
            {'color': (180, 80, 220),  'minerals': 4, 'radius': 22, 'score_val': 4},
            {'color': (100, 100, 255), 'minerals': 5, 'radius': 25, 'score_val': 5},
        ]
        
        # Fonts
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 40, bold=True)

        # Game Parameters
        self.MAX_STEPS = 1000
        self.TIMER_START_SECONDS = 20
        self.WIN_MINERALS = 50
        self.PLAYER_ACCEL = 0.4
        self.PLAYER_BRAKE = 0.9
        self.PLAYER_TURN_RATE = 0.1
        self.PLAYER_DRAG = 0.98
        self.PLAYER_MAX_SPEED = 7
        self.PLAYER_RADIUS = 10
        self.MINING_RANGE = 100
        self.MINING_RATE = 0.1  # minerals per frame
        self.INITIAL_ASTEROIDS = 8
        self.MAX_ASTEROIDS = 12
        
        # Initialize state variables
        self._np_random = None
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.steps = None
        self.score = None
        self.minerals_collected = None
        self.timer = None
        self.game_over = None
        self.win_condition_met = None
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.mining_target = None
        self.mining_beam_active = None
        self.mineral_mine_buffer = 0.0
        self.last_dist_to_asteroid = None
        
        self.reset()
        # self.validate_implementation() # Uncomment for debugging

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._np_random is None:
            self._np_random = np.random.default_rng(seed)

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.array([0, 0], dtype=np.float32)
        self.player_angle = -math.pi / 2  # Pointing up

        self.steps = 0
        self.score = 0
        self.minerals_collected = 0
        self.timer = self.TIMER_START_SECONDS * self.FPS
        self.game_over = False
        self.win_condition_met = False

        self.asteroids = []
        self.particles = []
        self.stars = self._create_stars(200)
        for _ in range(self.INITIAL_ASTEROIDS):
            self._spawn_asteroid(initial_spawn=True)

        self.mining_target = None
        self.mining_beam_active = False
        self.mineral_mine_buffer = 0.0
        self.last_dist_to_asteroid = self._get_dist_to_nearest_asteroid()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.timer -= 1
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        self._update_player(movement)
        
        mining_reward = self._update_mining(space_held)
        reward += mining_reward

        self._update_asteroids()
        self._update_particles()
        
        # Proximity reward
        current_dist = self._get_dist_to_nearest_asteroid()
        if current_dist is not None and self.last_dist_to_asteroid is not None:
            dist_delta = self.last_dist_to_asteroid - current_dist
            reward += dist_delta * 0.01 # Small reward for getting closer
        self.last_dist_to_asteroid = current_dist

        # Collision check
        collision_reward = self._check_collisions()
        reward += collision_reward
        if collision_reward < 0:
            self.game_over = True
            # SFX: Explosion

        # Check win/loss conditions
        if self.minerals_collected >= self.WIN_MINERALS and not self.game_over:
            self.game_over = True
            self.win_condition_met = True
            reward += 100
            # SFX: Win Jingle
        
        if (self.timer <= 0 or self.steps >= self.MAX_STEPS) and not self.game_over:
            self.game_over = True
            reward -= 100
            # SFX: Loss Buzzer
        
        terminated = self.game_over
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement):
        # 1=up, 2=down, 3=left, 4=right
        if movement == 3: # Turn left
            self.player_angle -= self.PLAYER_TURN_RATE
        if movement == 4: # Turn right
            self.player_angle += self.PLAYER_TURN_RATE

        forward_vec = np.array([math.cos(self.player_angle), math.sin(self.player_angle)])
        
        if movement == 1: # Accelerate
            self.player_vel += forward_vec * self.PLAYER_ACCEL
        if movement == 2: # Brake
            self.player_vel *= self.PLAYER_BRAKE

        # Drag and speed limit
        self.player_vel *= self.PLAYER_DRAG
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel = (self.player_vel / speed) * self.PLAYER_MAX_SPEED
        
        self.player_pos += self.player_vel

        # Screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)

    def _update_mining(self, space_held):
        reward = 0
        self.mining_beam_active = False
        
        if not space_held:
            self.mining_target = None
            return reward

        # If we have a target, check if it's still valid
        if self.mining_target:
            dist = np.linalg.norm(self.player_pos - self.mining_target['pos'])
            if dist > self.MINING_RANGE or self.mining_target['minerals'] <= 0:
                self.mining_target = None

        # Find a new target if we don't have one
        if not self.mining_target:
            self.mining_target = self._get_closest_asteroid_in_range()

        # Mine the target
        if self.mining_target:
            self.mining_beam_active = True
            # SFX: Mining beam hum
            
            self.mining_target['minerals'] -= self.MINING_RATE
            self.mineral_mine_buffer += self.MINING_RATE

            if self.mineral_mine_buffer >= 1.0:
                minerals_gained = math.floor(self.mineral_mine_buffer)
                self.minerals_collected += minerals_gained
                reward += minerals_gained * 1.0 # +1 reward per mineral
                self.mineral_mine_buffer -= minerals_gained
                
            # Particle effect
            if self._np_random.random() < 0.5:
                self._spawn_particles(self.mining_target['pos'], 1, self.COLOR_MINERAL, 0.5)
        
        return reward
        
    def _update_asteroids(self):
        # Remove depleted asteroids
        self.asteroids = [a for a in self.asteroids if a['minerals'] > 0]
        # Respawn new ones
        while len(self.asteroids) < self.MAX_ASTEROIDS:
            self._spawn_asteroid()

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _check_collisions(self):
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < self.PLAYER_RADIUS + asteroid['radius']:
                self._spawn_particles(self.player_pos, 30, self.COLOR_PLAYER, 3)
                return -5
        return 0

    def _get_dist_to_nearest_asteroid(self):
        minable_asteroids = [a for a in self.asteroids if a['minerals'] > 0]
        if not minable_asteroids:
            return None
        distances = [np.linalg.norm(self.player_pos - a['pos']) for a in minable_asteroids]
        return min(distances)

    def _get_closest_asteroid_in_range(self):
        minable_asteroids = [a for a in self.asteroids if a['minerals'] > 0]
        in_range_asteroids = []
        for a in minable_asteroids:
            dist = np.linalg.norm(self.player_pos - a['pos'])
            if dist <= self.MINING_RANGE:
                in_range_asteroids.append((dist, a))
        
        if not in_range_asteroids:
            return None
        
        return min(in_range_asteroids, key=lambda x: x[0])[1]

    def _spawn_asteroid(self, initial_spawn=False):
        asteroid_type = self._np_random.choice(self.ASTEROID_TYPES, p=[0.35, 0.3, 0.2, 0.1, 0.05])
        
        if initial_spawn:
            # Spawn away from center
            while True:
                pos = self._np_random.random(2) * [self.WIDTH, self.HEIGHT]
                if np.linalg.norm(pos - self.player_pos) > 150:
                    break
        else:
            # Spawn at edges
            edge = self._np_random.integers(4)
            if edge == 0: # Top
                pos = np.array([self._np_random.random() * self.WIDTH, -30])
            elif edge == 1: # Bottom
                pos = np.array([self._np_random.random() * self.WIDTH, self.HEIGHT + 30])
            elif edge == 2: # Left
                pos = np.array([-30, self._np_random.random() * self.HEIGHT])
            else: # Right
                pos = np.array([self.WIDTH + 30, self._np_random.random() * self.HEIGHT])

        self.asteroids.append({
            'pos': pos.astype(np.float32),
            'radius': asteroid_type['radius'],
            'minerals': asteroid_type['minerals'],
            'color': asteroid_type['color'],
            'vertices': self._create_asteroid_shape(asteroid_type['radius'])
        })

    def _create_asteroid_shape(self, radius):
        num_vertices = self._np_random.integers(8, 13)
        vertices = []
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            dist = radius + self._np_random.uniform(-0.2, 0.2) * radius
            vertices.append((math.cos(angle) * dist, math.sin(angle) * dist))
        return vertices

    def _spawn_particles(self, pos, count, color, speed_mult):
        for _ in range(count):
            angle = self._np_random.random() * 2 * math.pi
            speed = self._np_random.random() * speed_mult + 0.5
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self._np_random.integers(20, 40),
                'color': color,
            })

    def _create_stars(self, count):
        return [
            (self._np_random.integers(self.WIDTH), 
             self._np_random.integers(self.HEIGHT), 
             self._np_random.choice([1, 2]),
             self._np_random.integers(50, 150))
            for _ in range(count)
        ]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for x, y, size, alpha in self.stars:
            color = (alpha, alpha, alpha)
            if size == 1:
                self.screen.set_at((x, y), color)
            else:
                pygame.draw.rect(self.screen, color, (x, y, 2, 2))

        # Asteroids
        for a in self.asteroids:
            points = [(v[0] + a['pos'][0], v[1] + a['pos'][1]) for v in a['vertices']]
            if len(points) > 2:
                pygame.gfxdraw.filled_polygon(self.screen, points, a['color'])
                pygame.gfxdraw.aapolygon(self.screen, points, a['color'])

        # Mining Beam
        if self.mining_beam_active and self.mining_target:
            start_pos = tuple(self.player_pos.astype(int))
            end_pos = tuple(self.mining_target['pos'].astype(int))
            pygame.draw.aaline(self.screen, self.COLOR_BEAM, start_pos, end_pos, 1)

        # Particles
        for p in self.particles:
            size = max(1, int(p['life'] / 10))
            pygame.draw.rect(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1]), size, size))

        # Player
        if not (self.game_over and self.minerals_collected < self.WIN_MINERALS):
            p1 = (self.player_pos[0] + math.cos(self.player_angle) * self.PLAYER_RADIUS,
                  self.player_pos[1] + math.sin(self.player_angle) * self.PLAYER_RADIUS)
            p2 = (self.player_pos[0] + math.cos(self.player_angle + 2.2) * self.PLAYER_RADIUS,
                  self.player_pos[1] + math.sin(self.player_angle + 2.2) * self.PLAYER_RADIUS)
            p3 = (self.player_pos[0] + math.cos(self.player_angle - 2.2) * self.PLAYER_RADIUS,
                  self.player_pos[1] + math.sin(self.player_angle - 2.2) * self.PLAYER_RADIUS)
            
            # Glow
            pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_PLAYER_GLOW)
            # Ship
            pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_PLAYER)
            pygame.gfxdraw.aatrigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_PLAYER)

    def _render_ui(self):
        # Minerals
        mineral_text = self.font_ui.render(f"MINERALS: {self.minerals_collected}/{self.WIN_MINERALS}", True, self.COLOR_MINERAL)
        self.screen.blit(mineral_text, (10, 10))

        # Timer
        time_left_sec = self.timer / self.FPS
        timer_str = f"TIME: {time_left_sec:.1f}"
        if time_left_sec < self.TIMER_START_SECONDS / 4:
            timer_color = self.COLOR_TIMER_CRITICAL
        elif time_left_sec < self.TIMER_START_SECONDS / 2:
            timer_color = self.COLOR_TIMER_WARN
        else:
            timer_color = self.COLOR_TIMER_NORMAL
        
        timer_text = self.font_ui.render(timer_str, True, timer_color)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            if self.win_condition_met:
                msg = "MISSION COMPLETE"
                color = self.COLOR_PLAYER
            else:
                msg = "MISSION FAILED"
                color = self.COLOR_TIMER_CRITICAL
            
            msg_text = self.font_msg.render(msg, True, color)
            msg_rect = msg_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            pygame.draw.rect(self.screen, (0,0,0,180), msg_rect.inflate(20, 20))
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "minerals": self.minerals_collected,
            "time_left": self.timer
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
    # Set `render_mode` to "human" to see the pygame window
    # Note: The official environment is headless ("rgb_array")
    
    # Monkey-patch the render method for human playback
    def render(self):
        if not hasattr(self, 'human_screen'):
            pygame.display.init()
            self.human_screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Asteroid Miner")
        
        # Get the observation frame
        obs_frame = self._get_observation()
        # The observation is (H, W, C), but pygame surfaces are (W, H)
        # So we need to transpose it back
        surf = pygame.surfarray.make_surface(np.transpose(obs_frame, (1, 0, 2)))
        self.human_screen.blit(surf, (0, 0))
        pygame.display.flip()

    GameEnv.render = render

    env = GameEnv()
    env.reset()
    
    terminated = False
    total_reward = 0
    
    print("Starting manual control test.")
    print(GameEnv.user_guide)
    
    # Game loop
    while not terminated:
        # Map keyboard keys to actions
        keys = pygame.key.get_pressed()
        
        mov_action = 0 # no-op
        if keys[pygame.K_UP]:
            mov_action = 1
        elif keys[pygame.K_DOWN]:
            mov_action = 2
        elif keys[pygame.K_LEFT]:
            mov_action = 3
        elif keys[pygame.K_RIGHT]:
            mov_action = 4

        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [mov_action, space_action, shift_action]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        env.render()
        
        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
    
    print(f"Game over. Final Score: {total_reward:.2f}")
    print(f"Minerals collected: {info['minerals']}")
    
    env.close()