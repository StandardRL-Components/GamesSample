
# Generated: 2025-08-27T15:00:35.894398
# Source Brief: brief_00858.md
# Brief Index: 858

        
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
        "Controls: ↑↓←→ to aim your jump. Hold space for a long jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hop across procedurally generated asteroids to reach the end goal in this top-down arcade space hopper."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_TIME = 60  # seconds
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_PLAYER_GLOW = (50, 255, 150, 50)
    COLOR_ASTEROID = (120, 130, 140)
    COLOR_ASTEROID_LINE = (100, 110, 120)
    COLOR_GOAL = (255, 80, 80)
    COLOR_GOAL_GLOW = (255, 80, 80, 70)
    COLOR_STAR = (200, 200, 220)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (20, 20, 20)

    # Game Physics
    SHORT_JUMP_FORCE = 6.0
    LONG_JUMP_FORCE = 10.0
    DRAG = 0.995
    NUM_ASTEROIDS = 25
    NUM_STARS = 150
    PLAYER_RADIUS = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_large = pygame.font.SysFont('monospace', 24, bold=True)
        self.font_small = pygame.font.SysFont('monospace', 16, bold=True)
        
        # Initialize state variables
        self.np_random = None
        self.player_pos = None
        self.player_vel = None
        self.is_grounded = None
        self.current_asteroid_idx = None
        self.last_asteroid_idx = None
        self.asteroids = None
        self.goal_asteroid_idx = None
        self.stars = None
        self.particles = None
        self.steps = None
        self.score = None
        self.timer = None
        self.game_over = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_TIME
        
        self._generate_stars()
        self._generate_level()

        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.timer -= 1 / self.FPS
        
        reward = self._handle_action_and_physics(action)
        self.score += reward

        self._update_particles()
        
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self._is_player_on_goal():
                # Victory reward
                time_bonus = 10 * (max(0, self.timer) / self.MAX_TIME)
                reward += 10 + time_bonus
            elif self._is_player_out_of_bounds():
                # Penalty for falling into space
                reward += -10

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action_and_physics(self, action):
        movement = action[0]
        space_held = action[1] == 1
        
        # Handle jumping action
        if self.is_grounded and movement != 0:
            self.is_grounded = False
            self.last_asteroid_idx = self.current_asteroid_idx
            self.current_asteroid_idx = -1

            jump_force = self.LONG_JUMP_FORCE if space_held else self.SHORT_JUMP_FORCE
            direction = pygame.math.Vector2(0, 0)
            if movement == 1: direction.y = -1 # Up
            elif movement == 2: direction.y = 1 # Down
            elif movement == 3: direction.x = -1 # Left
            elif movement == 4: direction.x = 1 # Right
            
            if direction.length() > 0:
                self.player_vel = direction.normalize() * jump_force
                # sfx: jump
                self._spawn_jump_particles(direction)

        # Update physics
        if not self.is_grounded:
            self.player_pos += self.player_vel
            self.player_vel *= self.DRAG
        
        # Check for landings
        reward = 0
        if not self.is_grounded:
            for i, asteroid in enumerate(self.asteroids):
                dist = self.player_pos.distance_to(asteroid['pos'])
                if dist < asteroid['radius'] + self.PLAYER_RADIUS:
                    self.is_grounded = True
                    self.player_vel = pygame.math.Vector2(0, 0)
                    # Snap to surface
                    angle_to_player = math.atan2(self.player_pos.y - asteroid['pos'].y, self.player_pos.x - asteroid['pos'].x)
                    self.player_pos = asteroid['pos'] + pygame.math.Vector2(math.cos(angle_to_player), math.sin(angle_to_player)) * (asteroid['radius'] + self.PLAYER_RADIUS)
                    
                    self.current_asteroid_idx = i
                    # sfx: land
                    self._spawn_land_particles()
                    
                    if i != self.last_asteroid_idx and i != self.goal_asteroid_idx:
                        reward = 0.1 # Reward for landing on a new asteroid
                    break
        return reward

    def _generate_stars(self):
        self.stars = []
        for _ in range(self.NUM_STARS):
            self.stars.append({
                'pos': pygame.math.Vector2(self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)),
                'size': self.np_random.random() * 1.5
            })

    def _generate_level(self):
        self.asteroids = []
        
        # Place start asteroid
        start_radius = self.np_random.integers(25, 35)
        start_pos = pygame.math.Vector2(self.np_random.integers(50, 100), self.HEIGHT / 2)
        self.asteroids.append({'pos': start_pos, 'radius': start_radius, 'angle': 0, 'rot_speed': (self.np_random.random() - 0.5) * 0.02})
        
        # Place goal asteroid
        goal_radius = self.np_random.integers(25, 35)
        goal_pos = pygame.math.Vector2(self.np_random.integers(self.WIDTH - 100, self.WIDTH - 50), self.HEIGHT / 2)
        
        # Generate intermediate asteroids
        for _ in range(self.NUM_ASTEROIDS):
            while True:
                radius = self.np_random.integers(15, 30)
                pos = pygame.math.Vector2(self.np_random.integers(20, self.WIDTH - 20), self.np_random.integers(20, self.HEIGHT - 20))
                
                # Check for overlap with existing asteroids
                is_overlapping = False
                for ast in self.asteroids:
                    if pos.distance_to(ast['pos']) < ast['radius'] + radius + 30: # 30px buffer
                        is_overlapping = True
                        break
                if pos.distance_to(goal_pos) < goal_radius + radius + 30:
                    is_overlapping = True

                if not is_overlapping:
                    self.asteroids.append({'pos': pos, 'radius': radius, 'angle': 0, 'rot_speed': (self.np_random.random() - 0.5) * 0.04})
                    break
        
        # Add goal asteroid to list and set index
        self.asteroids.append({'pos': goal_pos, 'radius': goal_radius, 'angle': 0, 'rot_speed': (self.np_random.random() - 0.5) * 0.02})
        self.goal_asteroid_idx = len(self.asteroids) - 1

        # Set player state
        self.player_pos = pygame.math.Vector2(start_pos.x, start_pos.y - start_radius - self.PLAYER_RADIUS)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.is_grounded = True
        self.current_asteroid_idx = 0
        self.last_asteroid_idx = 0

    def _is_player_on_goal(self):
        return self.is_grounded and self.current_asteroid_idx == self.goal_asteroid_idx

    def _is_player_out_of_bounds(self):
        return not (0 < self.player_pos.x < self.WIDTH and 0 < self.player_pos.y < self.HEIGHT)

    def _check_termination(self):
        if self._is_player_on_goal():
            return True
        if self._is_player_out_of_bounds():
            return True
        if self.timer <= 0:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_text(self, text, font, pos, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW):
        shadow_surface = font.render(text, True, shadow_color)
        self.screen.blit(shadow_surface, (pos[0] + 2, pos[1] + 2))
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

    def _render_background(self):
        for star in self.stars:
            size = max(0, int(star['size']))
            if size > 0:
                pygame.draw.circle(self.screen, self.COLOR_STAR, (int(star['pos'].x), int(star['pos'].y)), size)

    def _render_game(self):
        # Draw asteroids
        for i, asteroid in enumerate(self.asteroids):
            pos_i = (int(asteroid['pos'].x), int(asteroid['pos'].y))
            radius_i = int(asteroid['radius'])
            color = self.COLOR_GOAL if i == self.goal_asteroid_idx else self.COLOR_ASTEROID
            
            # Pulsating glow for goal
            if i == self.goal_asteroid_idx:
                glow_radius = radius_i + 8 + 4 * math.sin(self.steps * 0.1)
                pygame.gfxdraw.filled_circle(self.screen, pos_i[0], pos_i[1], int(glow_radius), self.COLOR_GOAL_GLOW)
                pygame.gfxdraw.aacircle(self.screen, pos_i[0], pos_i[1], int(glow_radius), self.COLOR_GOAL_GLOW)
            
            pygame.gfxdraw.filled_circle(self.screen, pos_i[0], pos_i[1], radius_i, color)
            pygame.gfxdraw.aacircle(self.screen, pos_i[0], pos_i[1], radius_i, color)
            
            # Rotation line
            asteroid['angle'] += asteroid['rot_speed']
            end_x = pos_i[0] + radius_i * math.cos(asteroid['angle'])
            end_y = pos_i[1] + radius_i * math.sin(asteroid['angle'])
            pygame.draw.aaline(self.screen, self.COLOR_ASTEROID_LINE, pos_i, (end_x, end_y))

        # Draw particles
        for p in self.particles:
            p_pos_i = (int(p['pos'].x), int(p['pos'].y))
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'][:3] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, p_pos_i[0], p_pos_i[1], int(p['size']), color)

        # Draw player
        player_pos_i = (int(self.player_pos.x), int(self.player_pos.y))
        
        # Player glow
        glow_radius = self.PLAYER_RADIUS + 5
        if not self.is_grounded: # More glow when in air
            glow_radius += 3 * (1 + math.sin(self.steps * 0.2))
        pygame.gfxdraw.filled_circle(self.screen, player_pos_i[0], player_pos_i[1], int(glow_radius), self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, player_pos_i[0], player_pos_i[1], int(glow_radius), self.COLOR_PLAYER_GLOW)

        # Player body
        pygame.gfxdraw.filled_circle(self.screen, player_pos_i[0], player_pos_i[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_pos_i[0], player_pos_i[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_ui(self):
        # Render timer
        time_str = f"TIME: {max(0, self.timer):.1f}"
        self._render_text(time_str, self.font_large, (10, 10))

        # Render score
        score_str = f"SCORE: {self.score:.1f}"
        score_size = self.font_large.size(score_str)
        self._render_text(score_str, self.font_large, (self.WIDTH - score_size[0] - 10, 10))

        if self.game_over:
            msg = ""
            if self._is_player_on_goal():
                msg = "GOAL REACHED!"
            elif self.timer <= 0:
                msg = "TIME UP!"
            elif self._is_player_out_of_bounds():
                msg = "LOST IN SPACE!"
            else:
                msg = "GAME OVER"
            
            msg_size = self.font_large.size(msg)
            self._render_text(msg, self.font_large, (self.WIDTH/2 - msg_size[0]/2, self.HEIGHT/2 - msg_size[1]/2))

    def _spawn_jump_particles(self, jump_direction):
        # sfx: woosh
        for _ in range(10):
            self.particles.append({
                'pos': self.player_pos.copy(),
                'vel': -jump_direction.normalize() * (1 + self.np_random.random() * 2) + pygame.math.Vector2(self.np_random.random() - 0.5, self.np_random.random() - 0.5),
                'life': 15 + self.np_random.integers(0, 10),
                'max_life': 25,
                'size': self.np_random.random() * 2 + 1,
                'color': self.COLOR_PLAYER
            })

    def _spawn_land_particles(self):
        # sfx: thud
        for i in range(20):
            angle = (i / 20) * 2 * math.pi
            self.particles.append({
                'pos': self.player_pos.copy(),
                'vel': pygame.math.Vector2(math.cos(angle), math.sin(angle)) * (1 + self.np_random.random()),
                'life': 10 + self.np_random.integers(0, 5),
                'max_life': 15,
                'size': self.np_random.random() * 1.5 + 1,
                'color': self.COLOR_ASTEROID
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['size'] *= 0.95

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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Space Hopper")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Action mapping for human play ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0 # unused

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        
        action = [movement, space_held, shift_held]

        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}. Score: {info['score']:.2f}")
            # Wait for a moment then reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)
        
    pygame.quit()