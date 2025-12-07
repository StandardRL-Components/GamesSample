import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:58:53.974288
# Source Brief: brief_00746.md
# Brief Index: 746
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Control a flock of four birds, adjusting their individual speeds to navigate through a field of obstacles. "
        "Maintain formation to maximize your score."
    )
    user_guide = (
        "Use ↑, ↓, ←, → to select and accelerate one of the four birds. "
        "Hold Shift with an arrow key to decelerate the selected bird."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60 # Visual FPS
        self.LOGIC_FPS = 30 # Physics/logic update rate
        self.MAX_STEPS = 60 * self.LOGIC_FPS # 60 seconds

        # Colors
        self.COLOR_BG = (135, 206, 235) # Sky Blue
        self.COLOR_OBSTACLE = (105, 105, 105) # Dark Gray
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_BG = (0, 0, 0, 128)
        self.BIRD_COLORS = [
            (255, 87, 34),   # Red/Orange
            (76, 175, 80),   # Green
            (33, 150, 243),  # Blue
            (255, 235, 59)   # Yellow
        ]
        self.COLOR_LINE_GOOD = (0, 255, 0, 150)
        self.COLOR_LINE_BAD = (255, 0, 0, 150)
        
        # Bird settings
        self.BIRD_RADIUS = 12
        self.BIRD_ACCELERATION = 0.1
        self.BIRD_MAX_SPEED = 5.0
        self.BIRD_MIN_SPEED = 0.5
        self.BIRD_COLLISION_PENALTY = 0.9 # Speed multiplier on collision
        self.IDEAL_DISTANCE = 100
        self.DISTANCE_MARGIN = 0.25 # 25% margin

        # Obstacle settings
        self.OBSTACLE_WIDTH = 40
        self.OBSTACLE_MIN_HEIGHT = 80
        self.OBSTACLE_MAX_HEIGHT = 200
        self.OBSTACLE_SPAWN_RATE = 1.5 # seconds
        self.OBSTACLE_INITIAL_SPEED = 2.0
        self.OBSTACLE_SPEED_INCREASE = 0.1 # Per 10 seconds

        # Win/Loss conditions
        self.WIN_OBSTACLES_AVOIDED = 20

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('sans-serif', 24, bold=True)
        self.font_score = pygame.font.SysFont('sans-serif', 32, bold=True)
        
        # Initialize state variables
        self.birds = []
        self.obstacles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.obstacles_avoided = 0
        self.obstacle_speed = 0
        self.time_since_last_spawn = 0
        
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.obstacles_avoided = 0
        self.obstacle_speed = self.OBSTACLE_INITIAL_SPEED
        self.time_since_last_spawn = 0.0

        # Initialize birds in a diamond formation
        self.birds = [
            {'pos': pygame.Vector2(self.WIDTH * 0.2, self.HEIGHT * 0.5), 'speed': 2.0, 'color': self.BIRD_COLORS[0]}, # Center-left
            {'pos': pygame.Vector2(self.WIDTH * 0.3, self.HEIGHT * 0.5), 'speed': 2.0, 'color': self.BIRD_COLORS[1]}, # Center-right
            {'pos': pygame.Vector2(self.WIDTH * 0.25, self.HEIGHT * 0.3), 'speed': 2.0, 'color': self.BIRD_COLORS[2]}, # Top
            {'pos': pygame.Vector2(self.WIDTH * 0.25, self.HEIGHT * 0.7), 'speed': 2.0, 'color': self.BIRD_COLORS[3]}, # Bottom
        ]
        
        self.obstacles = []
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held = action[1] == 1  # Not used in this design
        shift_held = action[2] == 1  # Used as a speed modifier (decrease)

        self._handle_input(movement, shift_held)
        
        # Update game logic
        self.steps += 1
        self._update_birds()
        self._update_obstacles()
        self._update_particles()

        reward = 0
        
        # Calculate rewards and check for collisions
        collision_reward, bird_collision = self._handle_collisions()
        reward += collision_reward
        
        formation_reward = self._calculate_formation_reward()
        reward += formation_reward
        
        self.score += reward

        # Check for termination conditions
        terminated = self._check_termination(bird_collision)
        if terminated:
            self.game_over = True
            if self.obstacles_avoided >= self.WIN_OBSTACLES_AVOIDED:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Loss penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, shift_held):
        speed_change = -self.BIRD_ACCELERATION if shift_held else self.BIRD_ACCELERATION
        
        bird_to_affect = -1
        if movement == 1: bird_to_affect = 0 # Brief says 'up' for bird 1
        elif movement == 2: bird_to_affect = 1 # Brief says 'down' for bird 2
        elif movement == 3: bird_to_affect = 2 # Brief says 'left' for bird 3
        elif movement == 4: bird_to_affect = 3 # Brief says 'right' for bird 4

        if bird_to_affect != -1:
            bird = self.birds[bird_to_affect]
            bird['speed'] = np.clip(bird['speed'] + speed_change, self.BIRD_MIN_SPEED, self.BIRD_MAX_SPEED)
            # Add particle effect for feedback
            # Sound: whoosh.wav
            particle_vel = pygame.Vector2(-2, random.uniform(-1, 1))
            self._spawn_particles(bird['pos'], bird['color'], 5, particle_vel)

    def _update_birds(self):
        for bird in self.birds:
            bird['pos'].x += bird['speed']
            # Wrap around screen
            if bird['pos'].x > self.WIDTH + self.BIRD_RADIUS:
                bird['pos'].x = -self.BIRD_RADIUS
            if bird['pos'].x < -self.BIRD_RADIUS:
                bird['pos'].x = self.WIDTH + self.BIRD_RADIUS

    def _update_obstacles(self):
        # Increase difficulty over time
        if self.steps > 0 and self.steps % (10 * self.LOGIC_FPS) == 0:
            self.obstacle_speed += self.OBSTACLE_SPEED_INCREASE
        
        # Spawn new obstacles
        self.time_since_last_spawn += 1 / self.LOGIC_FPS
        if self.time_since_last_spawn > self.OBSTACLE_SPAWN_RATE:
            self.time_since_last_spawn = 0
            self._spawn_obstacle()

        # Move and check existing obstacles
        reward = 0
        for obstacle in self.obstacles[:]:
            obstacle['rect'].x -= self.obstacle_speed
            if obstacle['rect'].right < 0:
                self.obstacles.remove(obstacle)
            
            # Check for successful avoidance
            if not obstacle['scored'] and obstacle['rect'].right < min(b['pos'].x for b in self.birds):
                obstacle['scored'] = True
                self.obstacles_avoided += 1
                reward += 1.0 # Reward for avoiding
                # Sound: success.wav

    def _spawn_obstacle(self):
        height = self.np_random.integers(self.OBSTACLE_MIN_HEIGHT, self.OBSTACLE_MAX_HEIGHT)
        y_pos = self.np_random.integers(0, self.HEIGHT - height)
        rect = pygame.Rect(self.WIDTH, y_pos, self.OBSTACLE_WIDTH, height)
        self.obstacles.append({'rect': rect, 'scored': False, 'hit': False})

    def _handle_collisions(self):
        reward = 0
        bird_collision_detected = False

        # Bird-Obstacle collisions
        for bird in self.birds:
            bird_rect = pygame.Rect(bird['pos'].x - self.BIRD_RADIUS, bird['pos'].y - self.BIRD_RADIUS, self.BIRD_RADIUS * 2, self.BIRD_RADIUS * 2)
            for obstacle in self.obstacles:
                if not obstacle['hit'] and bird_rect.colliderect(obstacle['rect']):
                    obstacle['hit'] = True
                    reward -= 5.0 # Penalty for hitting obstacle
                    # Sound: impact.wav
                    self._spawn_particles(bird['pos'], (255, 255, 255), 15, pygame.Vector2(-1, 0))
                    # Slow down all birds on any collision
                    for b in self.birds:
                        b['speed'] *= self.BIRD_COLLISION_PENALTY

        # Bird-Bird collisions
        for i in range(len(self.birds)):
            for j in range(i + 1, len(self.birds)):
                bird1 = self.birds[i]
                bird2 = self.birds[j]
                distance = bird1['pos'].distance_to(bird2['pos'])
                if distance < self.BIRD_RADIUS * 2:
                    bird_collision_detected = True
        
        return reward, bird_collision_detected

    def _calculate_formation_reward(self):
        num_pairs = 0
        num_good_pairs = 0
        for i in range(len(self.birds)):
            for j in range(i + 1, len(self.birds)):
                num_pairs += 1
                dist = self.birds[i]['pos'].distance_to(self.birds[j]['pos'])
                lower_bound = self.IDEAL_DISTANCE * (1 - self.DISTANCE_MARGIN)
                upper_bound = self.IDEAL_DISTANCE * (1 + self.DISTANCE_MARGIN)
                if lower_bound < dist < upper_bound:
                    num_good_pairs += 1
        
        formation_quality = num_good_pairs / num_pairs if num_pairs > 0 else 1
        # Scale reward from -0.5 to +0.1
        return -0.5 + (0.6 * formation_quality)

    def _check_termination(self, bird_collision):
        if bird_collision:
            return True
        if self.obstacles_avoided >= self.WIN_OBSTACLES_AVOIDED:
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
        self._render_distance_lines()
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obstacle['rect'])
        self._render_particles()
        for i, bird in enumerate(self.birds):
            self._render_bird(bird, i)

    def _render_bird(self, bird, index):
        pos = (int(bird['pos'].x), int(bird['pos'].y))
        # Glow effect
        glow_radius = int(self.BIRD_RADIUS * 1.5)
        glow_color = bird['color'] + (60,) # Add alpha
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, glow_color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], glow_radius, glow_color)
        
        # Main body
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BIRD_RADIUS, bird['color'])
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BIRD_RADIUS, bird['color'])

        # "Eye" to show which bird is which
        eye_offset = pygame.Vector2(self.BIRD_RADIUS * 0.5, 0)
        eye_pos = bird['pos'] + eye_offset
        pygame.draw.circle(self.screen, (0,0,0), (int(eye_pos.x), int(eye_pos.y)), 3)

    def _render_distance_lines(self):
        for i in range(len(self.birds)):
            for j in range(i + 1, len(self.birds)):
                p1 = self.birds[i]['pos']
                p2 = self.birds[j]['pos']
                dist = p1.distance_to(p2)
                
                lower_bound = self.IDEAL_DISTANCE * (1 - self.DISTANCE_MARGIN)
                upper_bound = self.IDEAL_DISTANCE * (1 + self.DISTANCE_MARGIN)
                
                if lower_bound < dist < upper_bound:
                    color = self.COLOR_LINE_GOOD
                else:
                    color = self.COLOR_LINE_BAD
                
                pygame.draw.aaline(self.screen, color, p1, p2, 1)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                rect = pygame.Rect(p['pos'].x - size//2, p['pos'].y - size//2, size, size)
                pygame.draw.rect(self.screen, color, rect)

    def _spawn_particles(self, pos, color, count, base_vel):
        for _ in range(count):
            self.particles.append({
                'pos': pos.copy(),
                'vel': base_vel + pygame.Vector2(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)),
                'life': random.randint(10, 20),
                'max_life': 20,
                'color': color,
                'size': random.randint(2, 5)
            })

    def _render_ui(self):
        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.LOGIC_FPS)
        timer_text = f"Time: {time_left:.1f}s"
        self._draw_text(timer_text, (10, 10), self.font_ui)
        
        # Obstacles Avoided
        avoid_text = f"Avoided: {self.obstacles_avoided} / {self.WIN_OBSTACLES_AVOIDED}"
        text_surf = self.font_ui.render(avoid_text, True, self.COLOR_TEXT)
        self._draw_text(avoid_text, (self.WIDTH - text_surf.get_width() - 10, 10), self.font_ui)
        
        # Formation Score
        num_pairs = 0
        num_good_pairs = 0
        for i in range(len(self.birds)):
            for j in range(i + 1, len(self.birds)):
                num_pairs += 1
                dist = self.birds[i]['pos'].distance_to(self.birds[j]['pos'])
                if self.IDEAL_DISTANCE * (1-self.DISTANCE_MARGIN) < dist < self.IDEAL_DISTANCE * (1+self.DISTANCE_MARGIN):
                    num_good_pairs += 1
        
        formation_text = f"Formation: {num_good_pairs}/{num_pairs}"
        text_surf_formation = self.font_score.render(formation_text, True, self.COLOR_TEXT)
        self._draw_text(formation_text, (self.WIDTH/2 - text_surf_formation.get_width()/2, self.HEIGHT - 40), self.font_score)

    def _draw_text(self, text, pos, font):
        text_surface = font.render(text, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect(topleft=pos)
        
        bg_rect = text_rect.inflate(10, 5)
        bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        bg_surf.fill(self.COLOR_TEXT_BG)
        self.screen.blit(bg_surf, bg_rect)
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "obstacles_avoided": self.obstacles_avoided,
            "time_left": (self.MAX_STEPS - self.steps) / self.LOGIC_FPS,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This part is for human testing and visualization
    # It does not use the standard gym loop
    
    # Action mapping for human keyboard
    # Key: (movement, space, shift)
    key_map = {
        pygame.K_w: (1, 0, 0), pygame.K_s: (1, 0, 1), # Bird 1: W=faster, S=slower
        pygame.K_UP: (2, 0, 0), pygame.K_DOWN: (2, 0, 1), # Bird 2: Up=faster, Down=slower
        pygame.K_a: (3, 0, 0), pygame.K_d: (3, 0, 1), # Bird 3: A=faster, D=slower
        pygame.K_LEFT: (4, 0, 0), pygame.K_RIGHT: (4, 0, 1), # Bird 4: Left=faster, Right=slower
    }

    obs, info = env.reset()
    done = False
    running = True
    
    # Create a display for rendering
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Flock Formation")
    
    while running:
        action = [0, 0, 0] # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Continuous key presses
        keys = pygame.key.get_pressed()
        action_taken = False
        for key, act in key_map.items():
            if keys[key]:
                action = list(act)
                action_taken = True
                break # Prioritize first key in map
        
        if not action_taken:
            action = [0, 0, 0]

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Obstacles Avoided: {info['obstacles_avoided']}")
            obs, info = env.reset()

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()