
# Generated: 2025-08-27T18:10:27.955509
# Source Brief: brief_01752.md
# Brief Index: 1752

        
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
        "Controls: ↑↓ to move vertically. ←→ to slow down or speed up. Hold space for a major boost."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a snail across treacherous, procedurally generated paths. Collect glowing orbs for points, "
        "but avoid red obstacles and falling into the abyss. Complete 3 stages to win!"
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
        
        # Screen dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Colors
        self.COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
        self.COLOR_BG_BOTTOM = (70, 130, 180)  # Steel Blue
        self.COLOR_PATH = (34, 139, 34)  # Forest Green
        self.COLOR_PATH_EDGE = (0, 100, 0)  # Dark Green
        self.COLOR_SNAIL_BODY = (107, 142, 35)  # Olive Drab
        self.COLOR_SNAIL_SHELL = (210, 180, 140)  # Tan
        self.COLOR_SNAIL_EYE = (255, 255, 255)
        self.COLOR_ITEM = (255, 255, 0)  # Yellow
        self.COLOR_OBSTACLE = (220, 20, 60)  # Crimson
        self.COLOR_TEXT = (255, 255, 240)  # Ivory
        self.COLOR_PARTICLE_BOOST = (255, 165, 0) # Orange
        self.COLOR_PARTICLE_ITEM = (255, 215, 0) # Gold
        
        # Game constants
        self.FPS = 30
        self.SNAIL_SCREEN_X = 150
        self.SNAIL_BASE_SPEED_Y = 4
        self.BASE_SCROLL_SPEED = 2.0
        self.BOOST_MULTIPLIER = 2.5
        self.STAGE_LENGTH_PIXELS = self.SCREEN_WIDTH * 10 # 10 screens long
        self.MAX_EPISODE_STEPS = 5400 # 3 stages * 60s * 30fps
        
        # Initialize state variables
        self.snail_pos = None
        self.snail_vel_y = None
        self.camera_x = None
        self.path_nodes = None
        self.items = None
        self.obstacles = None
        self.particles = None
        self.lives = None
        self.score = None
        self.stage = None
        self.stage_difficulty_modifier = None
        self.time_steps_in_stage = None
        self.total_time_steps = None
        self.is_falling = None
        self.fall_timer = None
        self.game_over = None
        self.game_won = None
        
        # Initialize state
        self.reset()

        # Self-check
        self.validate_implementation()
    
    def _generate_stage(self):
        """Generates the path, items, and obstacles for the current stage."""
        self.path_nodes = []
        last_y = self.SCREEN_HEIGHT / 2
        
        # Generate path nodes
        num_nodes = int(self.STAGE_LENGTH_PIXELS / 50) # Node every 50px
        for i in range(num_nodes):
            x = i * 50
            y = last_y + self.np_random.uniform(-40, 40)
            y = np.clip(y, 80, self.SCREEN_HEIGHT - 80)
            width = self.np_random.uniform(80, 150) / self.stage_difficulty_modifier
            is_gap = self.np_random.random() < (0.05 * self.stage_difficulty_modifier)
            self.path_nodes.append({'x': x, 'y': y, 'width': width, 'is_gap': is_gap})
            last_y = y
        
        # Ensure start and end are not gaps
        self.path_nodes[0]['is_gap'] = False
        self.path_nodes[1]['is_gap'] = False
        self.path_nodes[-1]['is_gap'] = False
        self.path_nodes[-2]['is_gap'] = False

        # Generate items and obstacles
        self.items = []
        self.obstacles = []
        for i in range(1, len(self.path_nodes)):
            if not self.path_nodes[i]['is_gap']:
                path_y = self.path_nodes[i]['y']
                path_x = self.path_nodes[i]['x']
                
                # Add items
                if self.np_random.random() < 0.1:
                    item_y = path_y + self.np_random.uniform(-20, 20)
                    self.items.append(pygame.Vector2(path_x, item_y))

                # Add obstacles
                if self.np_random.random() < (0.08 * self.stage_difficulty_modifier):
                    self.obstacles.append(pygame.Vector2(path_x, path_y))

    def _get_path_info_at(self, world_x):
        """Gets path y, width, and gap status at a given world x-coordinate."""
        world_x = max(0, world_x)
        node_index = int(world_x / 50)
        
        if node_index + 1 >= len(self.path_nodes):
            # Past the end of the path
            p1 = self.path_nodes[-1]
            return p1['y'], p1['width'], p1['is_gap']
            
        p1 = self.path_nodes[node_index]
        p2 = self.path_nodes[node_index + 1]
        
        if p1['is_gap']:
            return p1['y'], 0, True

        # Interpolate between nodes
        interp_factor = (world_x % 50) / 50
        y = p1['y'] + (p2['y'] - p1['y']) * interp_factor
        width = p1['width'] + (p2['width'] - p1['width']) * interp_factor
        return y, width, False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.stage = 1
        self.stage_difficulty_modifier = 1.0
        self._generate_stage()
        
        path_start_y, _, _ = self._get_path_info_at(self.SNAIL_SCREEN_X)
        self.snail_pos = pygame.Vector2(self.SNAIL_SCREEN_X, path_start_y)
        self.snail_vel_y = 0
        
        self.camera_x = 0
        self.particles = []
        self.lives = 3
        self.score = 0
        self.total_time_steps = 0
        self.time_steps_in_stage = 0
        
        self.is_falling = False
        self.fall_timer = 0
        self.game_over = False
        self.game_won = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        # Unpack factorized action
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Update Timers ---
        self.total_time_steps += 1
        self.time_steps_in_stage += 1
        stage_time_limit = 60 * self.FPS
        
        # --- Handle Snail Movement ---
        scroll_speed = self.BASE_SCROLL_SPEED
        
        if not self.is_falling:
            # Vertical movement
            if movement == 1: self.snail_vel_y = -self.SNAIL_BASE_SPEED_Y
            elif movement == 2: self.snail_vel_y = self.SNAIL_BASE_SPEED_Y
            else: self.snail_vel_y = 0

            # Horizontal movement (scroll speed adjustment)
            if movement == 3: scroll_speed *= 0.7
            if movement == 4: scroll_speed *= 1.3
            if space_held: 
                scroll_speed *= self.BOOST_MULTIPLIER
                # Add boost particles
                if self.total_time_steps % 2 == 0:
                    self._add_particles(self.snail_pos - pygame.Vector2(15, 0), 10, self.COLOR_PARTICLE_BOOST, count=2)
        
        self.snail_pos.y += self.snail_vel_y
        self.snail_pos.y = np.clip(self.snail_pos.y, 0, self.SCREEN_HEIGHT)
        self.camera_x += scroll_speed

        # --- Collision and Game Logic ---
        snail_world_x = self.camera_x + self.snail_pos.x
        path_y, path_width, is_gap = self._get_path_info_at(snail_world_x)
        
        if self.is_falling:
            self.snail_pos.y += 10 # Gravity
            self.fall_timer -= 1
            if self.fall_timer <= 0:
                self.is_falling = False
                if self.lives > 0:
                    # Respawn
                    self.camera_x = max(0, self.camera_x - 200) # Move back a bit
                    new_snail_world_x = self.camera_x + self.SNAIL_SCREEN_X
                    path_y_respawn, _, _ = self._get_path_info_at(new_snail_world_x)
                    self.snail_pos.y = path_y_respawn
                else:
                    self.game_over = True
            
        else: # Not falling
            # Check if on path
            is_on_path = not is_gap and (abs(self.snail_pos.y - path_y) < path_width / 2)
            if is_on_path:
                reward += 0.01 # Small reward for surviving
            else:
                # Fell off path
                reward -= 5.0
                self.lives -= 1
                self.is_falling = True
                self.fall_timer = self.FPS # 1 second fall animation
                # sfx: fall_whistle.wav
            
            # Check item collision
            for item in self.items[:]:
                if self.snail_pos.distance_to(item - pygame.Vector2(self.camera_x, 0)) < 20:
                    self.items.remove(item)
                    self.score += 10
                    reward += 10.0
                    self._add_particles(self.snail_pos, 15, self.COLOR_PARTICLE_ITEM, count=20)
                    # sfx: item_collect.wav

            # Check obstacle collision
            for obstacle in self.obstacles[:]:
                if self.snail_pos.distance_to(obstacle - pygame.Vector2(self.camera_x, 0)) < 25:
                    self.obstacles.remove(obstacle)
                    self.score -= 5
                    reward -= 0.5
                    self.camera_x = max(0, self.camera_x - 30) # Knockback
                    self._add_particles(self.snail_pos, 10, self.COLOR_OBSTACLE, count=15)
                    # sfx: obstacle_hit.wav

        # --- Update Particles ---
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

        # --- Check for Stage/Game End ---
        terminated = False
        if self.camera_x >= self.STAGE_LENGTH_PIXELS:
            # Stage complete
            self.score += 50
            reward += 50.0
            self.stage += 1
            if self.stage > 3:
                self.game_won = True
                self.game_over = True
                reward += 100.0 # Bonus for winning
            else:
                # Advance to next stage
                self.stage_difficulty_modifier += 0.15
                self._generate_stage()
                self.camera_x = 0
                self.time_steps_in_stage = 0
                path_start_y, _, _ = self._get_path_info_at(self.SNAIL_SCREEN_X)
                self.snail_pos.y = path_start_y
                # sfx: stage_clear.wav
        
        if self.lives <= 0:
            self.game_over = True
        
        if self.time_steps_in_stage >= stage_time_limit:
            self.lives -= 1
            if self.lives <= 0:
                self.game_over = True
            else: # Reset stage on time out
                self.camera_x = 0
                self.time_steps_in_stage = 0
                path_start_y, _, _ = self._get_path_info_at(self.SNAIL_SCREEN_X)
                self.snail_pos.y = path_start_y


        if self.total_time_steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True

        if self.game_over:
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _add_particles(self, pos, radius, color, count=10):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': self.np_random.integers(10, 20),
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _render_background(self):
        """Draws a vertical gradient for the background."""
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_path(self):
        """Renders the scrolling path."""
        view_start_x = int(self.camera_x)
        view_end_x = view_start_x + self.SCREEN_WIDTH + 50 # Draw a bit extra
        
        start_node_idx = max(0, int(view_start_x / 50))
        end_node_idx = min(len(self.path_nodes) - 1, int(view_end_x / 50))

        for i in range(start_node_idx, end_node_idx):
            p1 = self.path_nodes[i]
            p2 = self.path_nodes[i+1]
            
            if p1['is_gap']:
                continue
                
            x1_screen = p1['x'] - self.camera_x
            x2_screen = p2['x'] - self.camera_x

            # Create polygon for the path segment
            poly_points = [
                (x1_screen, p1['y'] - p1['width']/2),
                (x2_screen, p2['y'] - p2['width']/2),
                (x2_screen, p2['y'] + p2['width']/2),
                (x1_screen, p1['y'] + p1['width']/2),
            ]
            
            # Use gfxdraw for anti-aliasing
            pygame.gfxdraw.aapolygon(self.screen, poly_points, self.COLOR_PATH_EDGE)
            pygame.gfxdraw.filled_polygon(self.screen, poly_points, self.COLOR_PATH)

    def _render_entities(self):
        """Renders items and obstacles."""
        # Render obstacles
        for obs in self.obstacles:
            screen_pos_x = obs.x - self.camera_x
            if 0 < screen_pos_x < self.SCREEN_WIDTH:
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos_x), int(obs.y), 12, self.COLOR_OBSTACLE)
                pygame.gfxdraw.aacircle(self.screen, int(screen_pos_x), int(obs.y), 12, (0,0,0))

        # Render items
        for item in self.items:
            screen_pos_x = item.x - self.camera_x
            if 0 < screen_pos_x < self.SCREEN_WIDTH:
                # Pulsing effect for items
                pulse = math.sin(self.total_time_steps * 0.2) * 2
                radius = int(8 + pulse)
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos_x), int(item.y), radius, self.COLOR_ITEM)
                pygame.gfxdraw.aacircle(self.screen, int(screen_pos_x), int(item.y), radius, (255,255,255,100))

    def _render_snail(self):
        """Renders the player's snail."""
        pos = self.snail_pos
        
        # Fall animation
        if self.is_falling:
            scale = self.fall_timer / self.FPS
            shell_radius = int(15 * scale)
            body_size = (int(25 * scale), int(12 * scale))
        else:
            shell_radius = 15
            body_size = (25, 12)
        
        if shell_radius <= 0: return # Don't draw if too small

        # Body
        body_rect = pygame.Rect(pos.x - body_size[0]/2, pos.y, body_size[0], body_size[1])
        pygame.draw.ellipse(self.screen, self.COLOR_SNAIL_BODY, body_rect)
        
        # Shell
        shell_pos = (int(pos.x), int(pos.y))
        pygame.gfxdraw.filled_circle(self.screen, shell_pos[0], shell_pos[1], shell_radius, self.COLOR_SNAIL_SHELL)
        pygame.gfxdraw.aacircle(self.screen, shell_pos[0], shell_pos[1], shell_radius, (0,0,0,50))
        
        # Eye
        eye_pos = (int(pos.x + 10), int(pos.y - 5))
        pygame.draw.circle(self.screen, self.COLOR_SNAIL_EYE, eye_pos, 5)
        pupil_offset_y = self.snail_vel_y * 0.2
        pygame.draw.circle(self.screen, (0,0,0), (eye_pos[0], int(eye_pos[1] + pupil_offset_y)), 2)
        
    def _render_particles(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['size']))

    def _render_ui(self):
        """Renders the score, lives, and time."""
        # Score
        score_surf = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 10, 10))
        
        # Stage
        stage_surf = self.font_ui.render(f"Stage: {self.stage}/3", True, self.COLOR_TEXT)
        self.screen.blit(stage_surf, (self.SCREEN_WIDTH / 2 - stage_surf.get_width() / 2, 10))
        
        # Lives
        lives_surf = self.font_ui.render(f"Lives: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_surf, (10, self.SCREEN_HEIGHT - lives_surf.get_height() - 10))
        
        # Time
        time_left = max(0, (60 * self.FPS - self.time_steps_in_stage) // self.FPS)
        time_surf = self.font_ui.render(f"Time: {time_left}", True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))
        
        if self.game_over:
            message = "YOU WIN!" if self.game_won else "GAME OVER"
            color = (0, 255, 0) if self.game_won else (255, 0, 0)
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            end_surf = self.font_game_over.render(message, True, color)
            self.screen.blit(end_surf, (self.SCREEN_WIDTH/2 - end_surf.get_width()/2, self.SCREEN_HEIGHT/2 - end_surf.get_height()/2))
    
    def _get_observation(self):
        self._render_background()
        self._render_path()
        self._render_entities()
        self._render_particles()
        self._render_snail()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.total_time_steps,
            "lives": self.lives,
            "stage": self.stage,
            "camera_x": self.camera_x
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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Snail Trail")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        movement = 0 # no-op
        space_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        
        action = [movement, space_held, 0] # Shift is unused

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame, just need to show it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Game Loop Control ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game...")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    env.close()