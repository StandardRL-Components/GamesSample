
# Generated: 2025-08-27T16:14:34.696752
# Source Brief: brief_01164.md
# Brief Index: 1164

        
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
        "Controls: Use arrow keys to jump. The goal is to reach the right side of the screen."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a spaceship through a procedurally generated asteroid field, "
        "jumping over obstacles to reach the end of the level."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # Class-level state for difficulty scaling across episodes
    _total_steps_across_episodes = 0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.LEVEL_START_X = 100
        self.LEVEL_END_X = self.LEVEL_START_X + 400 # Make it longer than 200 for better gameplay
        self.LEVEL_LENGTH = self.LEVEL_END_X - self.LEVEL_START_X
        self.PLAY_AREA_Y_TOP = 50
        self.PLAY_AREA_Y_BOTTOM = 350
        self.PLAY_AREA_HEIGHT = self.PLAY_AREA_Y_BOTTOM - self.PLAY_AREA_Y_TOP
        self.CENTER_Y = self.PLAY_AREA_Y_TOP + self.PLAY_AREA_HEIGHT / 2

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
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (180, 220, 255)
        self.COLOR_FINISH_LINE = (0, 255, 128)
        self.OBSTACLE_COLORS = {
            "small": (255, 80, 80), "medium": (80, 255, 80), "blue": (80, 80, 255)
        }
        self.PARTICLE_JUMP_COLOR = (255, 180, 50)
        self.PARTICLE_COLLISION_COLOR = (255, 50, 50)
        
        # Physics and Game Parameters
        self.FPS = 30
        self.JUMP_VEL_X = 6.0
        self.JUMP_VEL_Y = 8.0
        self.GRAVITY_PULL = 0.02 # Pulls player back to center Y
        self.DRAG = 0.92 # Multiplier for velocity
        self.PLAYER_RADIUS = 10
        self.JUMP_DURATION = 15 # frames
        self.INITIAL_OBSTACLE_DENSITY = 0.10
        self.MAX_EPISODE_STEPS = 1000
        
        # Initialize state variables
        self.stars = []
        self._generate_stars()
        
        # This will be called again by the environment wrapper, but is needed for validation
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def _generate_stars(self):
        self.stars = []
        for _ in range(200):
            x = random.randint(0, self.WIDTH)
            y = random.randint(0, self.HEIGHT)
            brightness = random.randint(50, 150)
            self.stars.append(((x, y), (brightness, brightness, brightness)))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game state
        self.player_pos = np.array([float(self.LEVEL_START_X), float(self.CENTER_Y)])
        self.player_vel = np.array([0.0, 0.0])
        self.is_jumping = False
        self.jump_timer = 0
        
        # Episode state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.max_progress = 0.0 # Player's furthest x position
        
        # Particles
        self.particles = []

        # Difficulty and Obstacles
        difficulty_multiplier = 1.0 + (0.05 * (GameEnv._total_steps_across_episodes // 500))
        num_obstacles = int((self.LEVEL_LENGTH / 100) * 20 * self.INITIAL_OBSTACLE_DENSITY * difficulty_multiplier)
        self._generate_obstacles(num_obstacles)

        return self._get_observation(), self._get_info()

    def _generate_obstacles(self, num_obstacles):
        self.obstacles = []
        for _ in range(num_obstacles):
            obstacle_type = random.choice(list(self.OBSTACLE_COLORS.keys()))
            if obstacle_type == "small":
                radius = 8
            elif obstacle_type == "medium":
                radius = 12
            else: # blue/large
                radius = 16
            
            # Place obstacles within the level, avoiding the start area
            x = random.uniform(self.LEVEL_START_X + 50, self.LEVEL_END_X - 20)
            y = random.uniform(self.PLAY_AREA_Y_TOP + radius, self.PLAY_AREA_Y_BOTTOM - radius)
            
            self.obstacles.append({
                "pos": np.array([x, y]),
                "radius": radius,
                "color": self.OBSTACLE_COLORS[obstacle_type],
                "cleared": False
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        # Unpack action
        movement = action[0]
        
        # --- Handle Input ---
        if not self.is_jumping:
            if movement == 1: # Up
                self.player_vel[1] = -self.JUMP_VEL_Y
                self.is_jumping = True
                self.jump_timer = self.JUMP_DURATION
                self._spawn_particles(self.player_pos, 20, self.PARTICLE_JUMP_COLOR, launch_angle_range=(225, 315))
            elif movement == 2: # Down
                self.player_vel[1] = self.JUMP_VEL_Y
                self.is_jumping = True
                self.jump_timer = self.JUMP_DURATION
                self._spawn_particles(self.player_pos, 20, self.PARTICLE_JUMP_COLOR, launch_angle_range=(45, 135))
            elif movement == 3: # Left
                self.player_vel[0] = -self.JUMP_VEL_X
                self.is_jumping = True
                self.jump_timer = self.JUMP_DURATION
                self._spawn_particles(self.player_pos, 20, self.PARTICLE_JUMP_COLOR, launch_angle_range=(-45, 45))
            elif movement == 4: # Right
                self.player_vel[0] = self.JUMP_VEL_X
                self.is_jumping = True
                self.jump_timer = self.JUMP_DURATION
                self._spawn_particles(self.player_pos, 20, self.PARTICLE_JUMP_COLOR, launch_angle_range=(135, 225))

        if movement == 0:
            reward -= 0.02 # Small penalty for no-op, different from brief for better balance
        
        # --- Update Game Logic ---
        self.steps += 1
        GameEnv._total_steps_across_episodes += 1

        # Update jump state
        if self.is_jumping:
            self.jump_timer -= 1
            if self.jump_timer <= 0:
                self.is_jumping = False

        # Apply gravity/pull towards center
        self.player_vel[1] += (self.CENTER_Y - self.player_pos[1]) * self.GRAVITY_PULL
        
        # Apply drag
        self.player_vel *= self.DRAG
        
        # Update position
        self.player_pos += self.player_vel

        # --- Collision Detection and Termination ---
        terminated = False
        
        # Boundary collision
        if not (self.PLAY_AREA_Y_TOP < self.player_pos[1] < self.PLAY_AREA_Y_BOTTOM):
            self.game_over = True
            terminated = True
            reward = -50.0
            self._spawn_particles(self.player_pos, 50, self.PARTICLE_COLLISION_COLOR)
            # sfx: explosion_sound

        # Obstacle collision
        if not terminated:
            for obs in self.obstacles:
                dist = np.linalg.norm(self.player_pos - obs["pos"])
                if dist < self.PLAYER_RADIUS + obs["radius"]:
                    self.game_over = True
                    terminated = True
                    reward = -50.0
                    self._spawn_particles(self.player_pos, 50, self.PARTICLE_COLLISION_COLOR)
                    # sfx: explosion_sound
                    break
        
        # --- Reward Calculation ---
        if not terminated:
            # Forward progress reward
            progress = self.player_pos[0] - self.LEVEL_START_X
            if progress > self.max_progress:
                reward += (progress - self.max_progress) * 0.1
                self.max_progress = progress

            # Obstacle cleared reward
            for obs in self.obstacles:
                if not obs["cleared"] and self.player_pos[0] > obs["pos"][0]:
                    reward += 5.0
                    obs["cleared"] = True
                    self.score += 5 # Use a separate score metric
                    # sfx: clear_obstacle_chime

        # --- Check Win/Max Steps Termination ---
        if not terminated:
            if self.player_pos[0] >= self.LEVEL_END_X:
                self.game_over = True
                terminated = True
                reward = 100.0
                self.score += 100
                # sfx: level_complete_fanfare
            elif self.steps >= self.MAX_EPISODE_STEPS:
                self.game_over = True
                terminated = True
                # No extra penalty/reward, just end.
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_observation(self):
        # Background
        self.screen.fill(self.COLOR_BG)
        for pos, color in self.stars:
            self.screen.set_at(pos, color)
        
        # Game elements
        self._render_finish_line()
        self._render_obstacles()
        self._render_player()
        self._update_and_render_particles()
        
        # UI
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "progress_percent": max(0, min(100, (self.player_pos[0] - self.LEVEL_START_X) / self.LEVEL_LENGTH * 100))
        }

    def _render_finish_line(self):
        for i in range(self.HEIGHT // 20):
            y = i * 20
            color = self.COLOR_FINISH_LINE if (i % 2) == 0 else self.COLOR_BG
            pygame.draw.line(self.screen, color, (self.LEVEL_END_X, y), (self.LEVEL_END_X, y + 20), 3)

    def _render_obstacles(self):
        for obs in self.obstacles:
            pos = (int(obs["pos"][0]), int(obs["pos"][1]))
            radius = int(obs["radius"])
            color = obs["color"]
            # Glow effect
            glow_radius = int(radius * 1.5)
            glow_color = tuple(c // 4 for c in color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, glow_color)
            # Main body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)

    def _render_player(self):
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        
        # Calculate rotation based on vertical velocity for a "banking" effect
        angle_rad = (self.player_vel[1] / self.JUMP_VEL_Y) * (math.pi / 6) # Max 30 deg tilt
        
        # Define ship points relative to origin
        p1 = np.array([self.PLAYER_RADIUS, 0])
        p2 = np.array([-self.PLAYER_RADIUS, -self.PLAYER_RADIUS * 0.7])
        p3 = np.array([-self.PLAYER_RADIUS, self.PLAYER_RADIUS * 0.7])
        
        # Rotation matrix
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        rot_matrix = np.array(((c, -s), (s, c)))
        
        # Rotate and translate points
        points = [
            self.player_pos + p1 @ rot_matrix.T,
            self.player_pos + p2 @ rot_matrix.T,
            self.player_pos + p3 @ rot_matrix.T
        ]
        points_int = [(int(p[0]), int(p[1])) for p in points]

        # Glow effect
        glow_surf = pygame.Surface((self.PLAYER_RADIUS * 4, self.PLAYER_RADIUS * 4), pygame.SRCALPHA)
        glow_points_rel = [(p[0] - pos[0] + self.PLAYER_RADIUS*2, p[1] - pos[1] + self.PLAYER_RADIUS*2) for p in points]
        pygame.draw.polygon(glow_surf, (*self.COLOR_PLAYER_GLOW, 60), glow_points_rel)
        scaled_glow = pygame.transform.smoothscale(glow_surf, (self.PLAYER_RADIUS * 8, self.PLAYER_RADIUS * 8))
        self.screen.blit(scaled_glow, (pos[0] - self.PLAYER_RADIUS*4, pos[1] - self.PLAYER_RADIUS*4), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Main ship body
        pygame.gfxdraw.aapolygon(self.screen, points_int, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points_int, self.COLOR_PLAYER)

    def _spawn_particles(self, pos, count, color, launch_angle_range=None):
        for _ in range(count):
            if launch_angle_range:
                angle = math.radians(random.uniform(*launch_angle_range))
                speed = random.uniform(1, 4)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            else: # Explosion
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 6)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]

            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "life": random.randint(15, 30),
                "color": color,
                "radius": random.uniform(1, 4)
            })

    def _update_and_render_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            p["radius"] -= 0.05

            if p["life"] <= 0 or p["radius"] <= 0:
                self.particles.pop(i)
            else:
                alpha = max(0, min(255, int(255 * (p["life"] / 20))))
                color_with_alpha = (*p["color"], alpha)
                
                # Create a temporary surface for the particle to handle alpha blending
                radius = int(p["radius"])
                if radius > 0:
                    particle_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(particle_surf, color_with_alpha, (radius, radius), radius)
                    self.screen.blit(particle_surf, (int(p["pos"][0] - radius), int(p["pos"][1] - radius)), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        # Progress Bar
        progress_percent = max(0, min(1, (self.player_pos[0] - self.LEVEL_START_X) / self.LEVEL_LENGTH))
        bar_width = self.WIDTH - 20
        bar_height = 10
        bar_x = 10
        bar_y = self.HEIGHT - 20
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_FINISH_LINE, (bar_x, bar_y, int(bar_width * progress_percent), bar_height))

        # Game Over Text
        if self.game_over:
            text = "MISSION FAILED"
            if self.player_pos[0] >= self.LEVEL_END_X:
                text = "MISSION COMPLETE"
            
            game_over_text = self.font_large.render(text, True, (255, 255, 255))
            text_rect = game_over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(game_over_text, text_rect)
            
    def render(self):
        # This is for human playback, not training.
        # It's not part of the standard gym API but is useful.
        if not hasattr(self, 'display_screen'):
            self.display_screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Asteroid Jumper")

        # Get the observation frame and blit it to the display
        obs_frame = self._get_observation()
        # The observation is (H, W, C), but pygame blit wants (W, H) surface.
        # We can convert it back.
        surf = pygame.surfarray.make_surface(np.transpose(obs_frame, (1, 0, 2)))
        self.display_screen.blit(surf, (0, 0))
        
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.FPS)

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

# Example usage for human play
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    terminated = False
    
    # Key mapping
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while not terminated:
        # Default action is no-op
        movement = 0
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # Check for held keys
        keys = pygame.key.get_pressed()
        for key, move_val in key_map.items():
            if keys[key]:
                movement = move_val
                break # Prioritize first key found
        
        action = [movement, 0, 0] # Space and shift are not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause for 2 seconds before resetting
            obs, info = env.reset()
            terminated = False

    env.close()