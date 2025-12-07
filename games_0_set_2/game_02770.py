
# Generated: 2025-08-27T21:23:31.274620
# Source Brief: brief_02770.md
# Brief Index: 2770

        
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
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold Shift to drift. Use boost with Space."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced isometric arcade racer. Drift through corners, collect boosts, "
        "and race against the clock to reach the finish line while dodging obstacles."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_GRID = (30, 35, 60)
    COLOR_TRACK = (50, 55, 80)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255, 50)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_POWERUP = (50, 255, 50)
    COLOR_FINISH_LINE = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    
    # Screen Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Game Parameters
    TIME_LIMIT_SECONDS = 60
    FPS = 30 # For physics calculations
    MAX_STEPS = TIME_LIMIT_SECONDS * FPS
    FINISH_LINE_Y = 15000
    TRACK_WIDTH = 600
    NUM_OBSTACLES = 40
    NUM_POWERUPS = 10

    # Player Physics
    ACCELERATION = 0.5
    BRAKING = 0.8
    FRICTION = 0.96
    MAX_SPEED = 12
    TURN_SPEED = 3.0
    DRIFT_TURN_SPEED = 5.0
    DRIFT_FRICTION = 0.98
    BOOST_MULTIPLIER = 2.0
    BOOST_DURATION = 3 * FPS # 3 seconds

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.ui_font = pygame.font.SysFont("monospace", 18, bold=True)
        self.game_over_font = pygame.font.SysFont("monospace", 40, bold=True)

        self.render_mode = render_mode
        self.game_objects = []
        self.particles = []
        
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.termination_reason = ""

        # Player state
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_angle = -90 # Pointing "up" in iso view
        self.boost_charges = 0
        self.boost_timer = 0
        
        # World state
        self.camera_pos = pygame.Vector2(0, 0)
        self.last_distance_to_finish = self.FINISH_LINE_Y
        self.obstacle_speed_modifier = 1.0

        # Generate world
        self._generate_world()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._update_player(movement, space_held, shift_held)
        self._update_world()
        
        reward, terminated = self._process_interactions_and_rewards()
        self.score += reward
        self.game_over = terminated
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True
            terminated = True
            reward -= 10 # Apply time limit penalty
            self.score -= 10
            self.termination_reason = "TIME LIMIT"
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_player(self, movement, space_held, shift_held):
        # Handle boost activation
        if space_held and self.boost_charges > 0 and self.boost_timer <= 0:
            self.boost_charges -= 1
            self.boost_timer = self.BOOST_DURATION
            # sfx: boost_activate.wav

        # Turning
        turn_speed = self.DRIFT_TURN_SPEED if shift_held else self.TURN_SPEED
        if movement == 3: # Left
            self.player_angle -= turn_speed
        if movement == 4: # Right
            self.player_angle += turn_speed

        # Acceleration/Braking
        acceleration_vector = pygame.Vector2(1, 0).rotate(self.player_angle)
        if movement == 1: # Up
            self.player_vel += acceleration_vector * self.ACCELERATION
        elif movement == 2: # Down
            self.player_vel *= self.BRAKING

        # Apply friction
        friction = self.DRIFT_FRICTION if shift_held else self.FRICTION
        self.player_vel *= friction

        # Speed limit & boost
        current_max_speed = self.MAX_SPEED
        if self.boost_timer > 0:
            current_max_speed *= self.BOOST_MULTIPLIER
            self.boost_timer -= 1
            # Add boost particles
            if self.steps % 2 == 0:
                self._emit_particles(self.player_pos, 5, self.COLOR_PLAYER, -acceleration_vector)

        if self.player_vel.length() > current_max_speed:
            self.player_vel.scale_to_length(current_max_speed)

        # Update position
        self.player_pos += self.player_vel

    def _update_world(self):
        # Update camera to follow player
        self.camera_pos.x = self.player_pos.x
        self.camera_pos.y = self.player_pos.y
        
        # Update obstacle speed
        if self.steps > 0 and self.steps % 1000 == 0:
            self.obstacle_speed_modifier += 0.05
        
        # Update dynamic obstacles
        for obj in self.game_objects:
            if obj['type'] == 'obstacle_h':
                obj['pos'].x = obj['base_pos'].x + math.sin(self.steps * 0.05 * self.obstacle_speed_modifier) * 150
            elif obj['type'] == 'obstacle_v':
                obj['pos'].y = obj['base_pos'].y + math.sin(self.steps * 0.05 * self.obstacle_speed_modifier) * 150
        
        # Update particles
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1

    def _process_interactions_and_rewards(self):
        reward = 0
        terminated = False

        # Reward for progress towards finish line
        distance_to_finish = max(0, self.FINISH_LINE_Y - self.player_pos.y)
        progress = self.last_distance_to_finish - distance_to_finish
        reward += progress * 0.1 # +0.1 for moving 1 unit closer
        self.last_distance_to_finish = distance_to_finish
        
        # Check for reaching finish line
        if self.player_pos.y >= self.FINISH_LINE_Y:
            reward += 100
            terminated = True
            self.termination_reason = "FINISH!"
            # sfx: win.wav
            return reward, terminated

        # Check collisions with game objects
        player_rect = pygame.Rect(self.player_pos.x - 5, self.player_pos.y - 5, 10, 10)
        
        for obj in self.game_objects[:]:
            obj_rect = pygame.Rect(obj['pos'].x - obj['size']/2, obj['pos'].y - obj['size']/2, obj['size'], obj['size'])
            if player_rect.colliderect(obj_rect):
                if 'obstacle' in obj['type']:
                    reward = -50
                    terminated = True
                    self.termination_reason = "CRASHED"
                    # sfx: crash.wav
                    self._emit_particles(self.player_pos, 50, self.COLOR_OBSTACLE, spread=360)
                    return reward, terminated
                elif obj['type'] == 'powerup':
                    reward += 5
                    self.boost_charges = min(3, self.boost_charges + 1)
                    self.game_objects.remove(obj)
                    # sfx: powerup_collect.wav
                    self._emit_particles(obj['pos'], 30, self.COLOR_POWERUP)

        return reward, terminated

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Create a sorted list of all drawable entities ---
        # Sorting by world y-coordinate ensures correct draw order (occlusion)
        player_entity = {
            'type': 'player', 'pos': self.player_pos, 'y': self.player_pos.y,
            'angle': self.player_angle, 'vel': self.player_vel
        }
        
        # Add a y-coordinate for sorting
        for obj in self.game_objects:
            obj['y'] = obj['pos'].y

        render_list = self.game_objects + [player_entity]
        render_list.sort(key=lambda e: e['y'])
        
        # --- Render elements ---
        self._render_background_grid()

        for entity in render_list:
            screen_pos = self._world_to_screen(entity['pos'])
            if 'obstacle' in entity['type']:
                self._draw_aa_rect(screen_pos, entity['size'] * 0.8, self.COLOR_OBSTACLE)
            elif entity['type'] == 'powerup':
                pulse = (math.sin(self.steps * 0.2) + 1) / 2
                radius = int(entity['size'] * 0.6 + pulse * 5)
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), radius, self.COLOR_POWERUP)
                pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y), radius, self.COLOR_POWERUP)
            elif entity['type'] == 'finish_line':
                 self._draw_finish_line(screen_pos)
            elif entity['type'] == 'player':
                self._render_player(screen_pos, entity['angle'], entity['vel'])

        # Render particles on top
        for p in self.particles:
            screen_pos = self._world_to_screen(p['pos'])
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = p['color'] + (alpha,)
            pygame.draw.circle(self.screen, color, screen_pos, p['size'])

    def _render_player(self, pos, angle, vel):
        # Car body
        size = 12
        points = [
            pygame.Vector2(size, 0),
            pygame.Vector2(-size * 0.7, size * 0.6),
            pygame.Vector2(-size * 0.4, 0),
            pygame.Vector2(-size * 0.7, -size * 0.6)
        ]
        
        # Rotate points
        rotated_points = [p.rotate(angle) for p in points]
        
        # Apply isometric shear to velocity trail
        drift_angle = 0
        if vel.length() > 1:
            drift_angle = self.player_vel.angle_to(pygame.Vector2(1, 0).rotate(angle))
        
        iso_vel = pygame.Vector2(vel.x - vel.y, (vel.x + vel.y) / 2)
        
        # Draw glow
        glow_points = [(pos + p) for p in rotated_points]
        pygame.gfxdraw.filled_polygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)

        # Draw main body
        body_points = [(pos + p) for p in rotated_points]
        pygame.gfxdraw.filled_polygon(self.screen, body_points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, body_points, self.COLOR_PLAYER)

    def _render_background_grid(self):
        # Draw track edges
        left_edge_w = -self.TRACK_WIDTH / 2
        right_edge_w = self.TRACK_WIDTH / 2
        
        for y_offset in range(-20, 21, 2):
            world_y = self.camera_pos.y + y_offset * 50
            
            p1_world = pygame.Vector2(left_edge_w, world_y)
            p2_world = pygame.Vector2(right_edge_w, world_y)
            
            p1_screen = self._world_to_screen(p1_world)
            p2_screen = self._world_to_screen(p2_world)
            
            # Draw track lines
            if int(world_y) % 200 == 0:
                 pygame.draw.line(self.screen, self.COLOR_TRACK, p1_screen, p2_screen, 2)
        
        for x_offset in [-1, 1]:
            for y_offset in range(-20, 21):
                p1_world = pygame.Vector2(x_offset * self.TRACK_WIDTH/2, self.camera_pos.y + y_offset * 50)
                p2_world = pygame.Vector2(x_offset * self.TRACK_WIDTH/2, self.camera_pos.y + (y_offset+1) * 50)
                p1_screen = self._world_to_screen(p1_world)
                p2_screen = self._world_to_screen(p2_world)
                pygame.draw.line(self.screen, self.COLOR_TRACK, p1_screen, p2_screen, 2)

    def _render_ui(self):
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = f"TIME: {max(0, time_left):.1f}"
        
        distance_left = max(0, self.FINISH_LINE_Y - self.player_pos.y)
        dist_text = f"DIST: {int(distance_left)}"
        
        boost_text = f"BOOST: {'■' * self.boost_charges}{'□' * (3 - self.boost_charges)}"
        
        self._draw_text(time_text, (10, 10))
        self._draw_text(dist_text, (self.SCREEN_WIDTH - 150, 10))
        self._draw_text(boost_text, (10, 35))
        
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            end_text = self.game_over_font.render(self.termination_reason, True, self.COLOR_FINISH_LINE)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20))
            self.screen.blit(end_text, text_rect)
            
            score_text = self.ui_font.render(f"Final Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
            score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 20))
            self.screen.blit(score_text, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance_to_finish": self.FINISH_LINE_Y - self.player_pos.y,
            "boost_charges": self.boost_charges,
        }

    # --- Helper Functions ---
    def _generate_world(self):
        self.game_objects = []
        
        # Finish Line
        self.game_objects.append({
            'type': 'finish_line', 'pos': pygame.Vector2(0, self.FINISH_LINE_Y), 'size': self.TRACK_WIDTH
        })

        # Obstacles and Powerups
        for i in range(1, (self.FINISH_LINE_Y // 400)):
            y_pos = i * 400
            # Add obstacles for this segment
            for _ in range(self.NUM_OBSTACLES // (self.FINISH_LINE_Y // 400)):
                x = self.np_random.uniform(-self.TRACK_WIDTH / 2.2, self.TRACK_WIDTH / 2.2)
                y = y_pos + self.np_random.uniform(-180, 180)
                pos = pygame.Vector2(x, y)
                
                obstacle_type = self.np_random.choice(['obstacle_s', 'obstacle_h', 'obstacle_v'])
                self.game_objects.append({
                    'type': obstacle_type, 'pos': pos.copy(), 'base_pos': pos.copy(), 'size': 30
                })
            
            # Add powerups for this segment
            for _ in range(self.NUM_POWERUPS // (self.FINISH_LINE_Y // 400)):
                x = self.np_random.uniform(-self.TRACK_WIDTH / 2.2, self.TRACK_WIDTH / 2.2)
                y = y_pos + self.np_random.uniform(-180, 180)
                self.game_objects.append({
                    'type': 'powerup', 'pos': pygame.Vector2(x, y), 'size': 20
                })

    def _world_to_screen(self, world_pos):
        # Relative to camera
        rel_pos = world_pos - self.camera_pos
        
        # Isometric projection
        iso_x = rel_pos.x - rel_pos.y
        iso_y = (rel_pos.x + rel_pos.y) / 2
        
        # Center on screen, with player offset
        screen_x = self.SCREEN_WIDTH / 2 + iso_x
        screen_y = self.SCREEN_HEIGHT * 0.75 + iso_y # Player appears lower on screen
        
        return pygame.Vector2(screen_x, screen_y)

    def _draw_text(self, text, pos):
        text_surface = self.ui_font.render(text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, pos)

    def _draw_aa_rect(self, center_pos, size, color):
        half_size = size / 2
        points = [
            (center_pos.x - half_size, center_pos.y),
            (center_pos.x, center_pos.y - half_size),
            (center_pos.x + half_size, center_pos.y),
            (center_pos.x, center_pos.y + half_size),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _draw_finish_line(self, screen_pos):
        for i in range(10):
            color = self.COLOR_FINISH_LINE if (i % 2) == (int(self.steps*0.2) % 2) else self.COLOR_TRACK
            start = screen_pos + pygame.Vector2(i * self.TRACK_WIDTH/10, 0)
            end = screen_pos + pygame.Vector2((i+1) * self.TRACK_WIDTH/10, 0)
            pygame.draw.line(self.screen, color, start, end, 10)

    def _emit_particles(self, pos, count, color, direction=None, spread=45):
        for _ in range(count):
            if direction:
                angle = direction.angle_to(pygame.Vector2(1,0)) + self.np_random.uniform(-spread/2, spread/2)
            else:
                angle = self.np_random.uniform(0, 360)
            
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(1,0).rotate(angle) * speed
            lifespan = self.np_random.integers(10, 25)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'color': color,
                'size': self.np_random.uniform(1, 4)
            })
            
    def validate_implementation(self):
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
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Racer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(GameEnv.FPS)

    pygame.quit()