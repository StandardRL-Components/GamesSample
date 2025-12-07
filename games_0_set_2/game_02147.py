
# Generated: 2025-08-28T03:53:27.985856
# Source Brief: brief_02147.md
# Brief Index: 2147

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An expert implementation of a side-scrolling snail racing game as a Gymnasium environment.
    This environment features high-quality procedural graphics, smooth animations,
    and engaging, responsive gameplay mechanics.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: ↑ to accelerate, ↓ to decelerate. Hold space for a speed boost and shift to brake hard."
    )

    # Short, user-facing description of the game
    game_description = (
        "Control a racing snail in a side-view race. Overtake your opponents, dodge obstacles, "
        "and manage your speed to win all three stages before time runs out."
    )

    # Frames auto-advance at 30fps for smooth real-time gameplay
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    TRACK_Y = 300
    TRACK_HEIGHT = 100
    MAX_STAGES = 3

    # Colors (Bright for interactive, desaturated for background)
    COLOR_SKY = (135, 206, 235)
    COLOR_BG_HILLS_1 = (128, 178, 149)
    COLOR_BG_HILLS_2 = (100, 150, 120)
    COLOR_TRACK = (118, 158, 77)
    COLOR_PLAYER = (255, 87, 51)
    COLOR_PLAYER_GLOW = (255, 150, 120, 100)
    COLOR_AI1 = (51, 153, 255)
    COLOR_AI2 = (255, 215, 0)
    COLOR_OBSTACLE = (50, 50, 50)
    COLOR_OBSTACLE_MARKER = (255, 255, 0)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (40, 40, 40)
    COLOR_UI_BG = (0, 0, 0, 128)

    # Physics
    ACCELERATION = 0.04
    DECELERATION = 0.08
    BOOST_POWER = 0.2
    BRAKE_POWER = 0.15
    MAX_SPEED = 5.0
    FRICTION = 0.99
    STUN_DURATION = 0.5 * FPS  # 0.5 seconds

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        # This will be seeded by the environment's reset method
        self.np_random = None

        # Initialize state variables
        self.reset()
        
        # self.validate_implementation() # Uncomment for self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.total_score = 0
        self.stage = 1
        self.win_game = False
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        """Initializes state for the current stage."""
        self.steps = 0
        self.stage_score = 0
        self.time_remaining = 60 * self.FPS
        self.game_over = False
        self.stage_complete_frames = -1

        self.finish_line_x = 2000 + 1000 * self.stage
        
        self.player = {'pos': np.array([100.0, self.TRACK_Y - 20]), 'speed': 0.0, 'stun_timer': 0, 'rank': 3, 'name': "YOU"}
        
        ai_base_speed = 1.8 + (self.stage - 1) * 0.4
        self.ais = [
            {'pos': np.array([80.0, self.TRACK_Y - 20]), 'speed': ai_base_speed, 'stun_timer': 0, 'color': self.COLOR_AI1, 'name': "AI 1"},
            {'pos': np.array([60.0, self.TRACK_Y - 20]), 'speed': ai_base_speed, 'stun_timer': 0, 'color': self.COLOR_AI2, 'name': "AI 2"}
        ]
        
        self.obstacles = self._generate_obstacles()
        self.particles = []
        self.camera_x = 0
        
        # Generate procedural background elements for variety
        self.bg_hills_1 = self._generate_hills(50, 100, 200, 0.5)
        self.bg_hills_2 = self._generate_hills(70, 120, 150, 0.7)
        self.clouds = self._generate_clouds()

    def _generate_obstacles(self):
        obstacles = []
        num_obstacles = 5 + self.stage * 3
        for i in range(num_obstacles):
            x = 600 + i * self.np_random.uniform(200, 400)
            if x < self.finish_line_x - 300:
                obstacles.append(pygame.Rect(x, self.TRACK_Y - 30, 20, 30))
        return obstacles

    def _generate_hills(self, min_h, max_h, period, parallax_factor):
        points = []
        for x in range(-self.WIDTH, self.finish_line_x + self.WIDTH, 10):
            height = self.np_random.uniform(min_h, max_h) + math.sin(x / period) * 20
            points.append({'x': x, 'height': height, 'parallax': parallax_factor})
        return points

    def _generate_clouds(self):
        clouds = []
        for _ in range(20):
            x = self.np_random.uniform(-self.WIDTH, self.finish_line_x + self.WIDTH)
            y = self.np_random.uniform(20, 100)
            size = self.np_random.uniform(40, 100)
            clouds.append({'x': x, 'y': y, 'size': size, 'parallax': self.np_random.uniform(0.1, 0.3)})
        return clouds

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        old_pos_x = self.player['pos'][0]
        all_snails = [self.player] + self.ais
        old_ranks = {snail['name']: rank for rank, snail in enumerate(sorted(all_snails, key=lambda s: s['pos'][0], reverse=True))}

        self._handle_input(action)
        self._update_snails()
        
        collision_penalty = self._handle_collisions(action)
        reward += collision_penalty
        
        self._update_particles()
        self._update_camera()
        
        self.time_remaining -= 1
        self.steps += 1

        # --- Reward Calculation ---
        distance_gain = self.player['pos'][0] - old_pos_x
        reward += distance_gain * 0.1

        if self.player['speed'] < 0.5 and self.player['stun_timer'] <= 0:
            reward -= 0.2
        
        new_ranks = {snail['name']: rank for rank, snail in enumerate(sorted(all_snails, key=lambda s: s['pos'][0], reverse=True))}
        if new_ranks[self.player['name']] < old_ranks[self.player['name']]:
            reward += 2 # Overtake reward
            # sfx: positive chime

        # --- Termination and Stage Progression ---
        terminated = False
        if self.stage_complete_frames > 0:
            self.stage_complete_frames -= 1
            if self.stage_complete_frames == 0:
                self.stage += 1
                if self.stage > self.MAX_STAGES:
                    self.game_over = True
                    self.win_game = True
                    terminated = True
                else:
                    self._setup_stage()

        elif self.player['pos'][0] >= self.finish_line_x:
            reward += 100
            self.total_score += self.stage_score + reward
            self.stage_score = 0
            self.stage_complete_frames = self.FPS * 2 # 2 second pause
            # sfx: stage complete fanfare

        if self.time_remaining <= 0 and self.stage_complete_frames < 0:
            self.game_over = True
            terminated = True
            reward -= 100
            # sfx: game over sad tune

        self.stage_score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if self.player['stun_timer'] > 0:
            return

        if movement == 1: # Up
            self.player['speed'] += self.ACCELERATION
        if movement == 2: # Down
            self.player['speed'] -= self.DECELERATION

        if space_held:
            self.player['speed'] += self.BOOST_POWER
            # sfx: boost whoosh
            if self.np_random.random() < 0.5:
                self._create_particles(self.player['pos'], 3, (255, 180, 50), 2, 4)
        if shift_held:
            self.player['speed'] -= self.BRAKE_POWER
            # sfx: brake screech

    def _update_snails(self):
        # Update Player
        p = self.player
        if p['stun_timer'] > 0:
            p['stun_timer'] -= 1
            p['speed'] = 0
        else:
            p['speed'] *= self.FRICTION
            p['speed'] = np.clip(p['speed'], 0, self.MAX_SPEED)
        p['pos'][0] += p['speed']
        
        if p['speed'] > 1.0 and self.steps % 3 == 0:
            self._create_particles(p['pos'] + np.array([-15, 10]), 1, (200, 200, 200), 1, 2, lifetime=10)

        # Update AIs
        ai_base_speed = 1.8 + (self.stage - 1) * 0.4
        for ai in self.ais:
            if ai['stun_timer'] > 0:
                ai['stun_timer'] -= 1
                ai['speed'] = 0
            else:
                # Simple AI: try to maintain a speed with some randomness, slow down for obstacles
                target_speed = ai_base_speed + self.np_random.uniform(-0.2, 0.2)
                
                # Look ahead for obstacles
                is_obstacle_ahead = False
                for obs in self.obstacles:
                    if obs.left > ai['pos'][0] and obs.left < ai['pos'][0] + 100:
                        is_obstacle_ahead = True
                        break
                
                if is_obstacle_ahead:
                    target_speed *= 0.5 # Slow down
                
                ai['speed'] = ai['speed'] * 0.95 + target_speed * 0.05
                ai['speed'] = np.clip(ai['speed'], 0, self.MAX_SPEED - 0.5)

            ai['pos'][0] += ai['speed']
            if ai['speed'] > 1.0 and self.steps % 4 == 0:
                self._create_particles(ai['pos'] + np.array([-15, 10]), 1, (200, 200, 200), 1, 2, lifetime=10)

    def _handle_collisions(self, action):
        reward = 0
        space_held = action[1] == 1
        
        all_snails = [self.player] + self.ais
        for snail in all_snails:
            snail_rect = pygame.Rect(snail['pos'][0] - 15, snail['pos'][1] - 15, 30, 30)
            for obs in self.obstacles:
                if snail_rect.colliderect(obs):
                    if snail['stun_timer'] <= 0:
                        snail['stun_timer'] = self.STUN_DURATION
                        # sfx: collision thud
                        self._create_particles(snail['pos'], 15, (255, 255, 0), 2, 5)
                        if snail is self.player:
                            reward -= 5
                            if space_held: # Penalty for boosting into an obstacle
                                reward -= 1
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['lifetime'] -= 1
            p['pos'] += p['vel']
            p['size'] -= 0.1

    def _update_camera(self):
        all_snails = [self.player] + self.ais
        leader_x = max(s['pos'][0] for s in all_snails)
        target_camera_x = leader_x - self.WIDTH * 0.4
        self.camera_x = self.camera_x * 0.9 + target_camera_x * 0.1

    def _get_observation(self):
        self.screen.fill(self.COLOR_SKY)
        self._render_background()
        self._render_track()
        self._render_game_elements()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.total_score + self.stage_score,
            "steps": self.steps,
            "stage": self.stage,
            "time_remaining": self.time_remaining / self.FPS,
        }

    def _render_background(self):
        # Clouds
        for cloud in self.clouds:
            cx = int(cloud['x'] - self.camera_x * cloud['parallax'])
            cy = int(cloud['y'])
            size = int(cloud['size'])
            if -size < cx < self.WIDTH + size:
                pygame.gfxdraw.filled_ellipse(self.screen, cx, cy, int(size/2), int(size/4), (255,255,255,150))
                pygame.gfxdraw.filled_ellipse(self.screen, cx-int(size/4), cy+int(size/8), int(size/3), int(size/5), (255,255,255,150))
                pygame.gfxdraw.filled_ellipse(self.screen, cx+int(size/4), cy+int(size/8), int(size/3), int(size/5), (255,255,255,150))

        # Hills
        for hill_set, color in [(self.bg_hills_2, self.COLOR_BG_HILLS_2), (self.bg_hills_1, self.COLOR_BG_HILLS_1)]:
            for hill in hill_set:
                hx = int(hill['x'] - self.camera_x * hill['parallax'])
                if -200 < hx < self.WIDTH + 200:
                    pygame.gfxdraw.filled_ellipse(self.screen, hx, self.TRACK_Y, 200, int(hill['height']), color)

    def _render_track(self):
        pygame.draw.rect(self.screen, self.COLOR_TRACK, (0, self.TRACK_Y, self.WIDTH, self.TRACK_HEIGHT))

    def _render_game_elements(self):
        # Finish Line
        finish_x = int(self.finish_line_x - self.camera_x)
        if -20 < finish_x < self.WIDTH:
            for i in range(10):
                color = (255, 255, 255) if i % 2 == 0 else (0, 0, 0)
                pygame.draw.rect(self.screen, color, (finish_x, self.TRACK_Y + i * (self.TRACK_HEIGHT/10), 20, self.TRACK_HEIGHT/10))

        # Obstacles
        for obs in self.obstacles:
            obs_rect = obs.move(-self.camera_x, 0)
            if obs_rect.right > 0 and obs_rect.left < self.WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs_rect, border_radius=3)
                if self.steps % self.FPS < self.FPS / 2: # Flashing marker
                    pygame.draw.circle(self.screen, self.COLOR_OBSTACLE_MARKER, (obs_rect.centerx, obs_rect.top - 8), 4)

        # Snails
        all_snails = sorted([self.player] + self.ais, key=lambda s: s['pos'][1])
        for snail in all_snails:
            color = self.COLOR_PLAYER if snail is self.player else snail['color']
            self._render_snail(snail['pos'] - np.array([self.camera_x, 0]), color, snail['speed'], self.steps, snail is self.player)

    def _render_snail(self, pos, color, speed, frame, is_player):
        x, y = int(pos[0]), int(pos[1])
        
        # Glow for player
        if is_player:
            glow_radius = int(25 + math.sin(frame * 0.1) * 3)
            pygame.gfxdraw.filled_circle(self.screen, x, y, glow_radius, self.COLOR_PLAYER_GLOW)
        
        # Simple bobbing animation
        bob = math.sin(frame * 0.2 + x) * 3
        
        # Body
        body_rect = pygame.Rect(x - 20, y + bob, 35, 12)
        pygame.draw.ellipse(self.screen, color, body_rect)
        
        # Shell
        shell_color = tuple(np.clip(np.array(color) * 0.7, 0, 255))
        pygame.gfxdraw.filled_circle(self.screen, x - 5, y - 10 + int(bob), 12, shell_color)
        pygame.gfxdraw.aacircle(self.screen, x - 5, y - 10 + int(bob), 12, (0,0,0,50))

        # Eyes
        eye_y = y - 2 + bob
        pygame.draw.circle(self.screen, (255, 255, 255), (x + 12, eye_y - 8), 4)
        pygame.draw.circle(self.screen, (0, 0, 0), (x + 13, eye_y - 8), 2)
        
        # Speed text
        speed_text = f"{speed:.1f}"
        self._render_text(speed_text, (x, y - 35), self.font_small, self.COLOR_TEXT)

    def _render_particles(self):
        for p in self.particles:
            if p['size'] > 0:
                pos = p['pos'] - np.array([self.camera_x, 0])
                pygame.draw.circle(self.screen, p['color'], pos.astype(int), int(p['size']))

    def _render_ui(self):
        # UI background panels
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, self.WIDTH, 40))
        
        # Stage
        self._render_text(f"Stage: {self.stage}/{self.MAX_STAGES}", (10, 10), self.font_medium, self.COLOR_TEXT, align="left")
        
        # Time
        time_str = f"Time: {int(self.time_remaining / self.FPS):02d}"
        self._render_text(time_str, (self.WIDTH - 10, 10), self.font_medium, self.COLOR_TEXT, align="right")

        # Score
        score_str = f"Score: {int(self.total_score + self.stage_score)}"
        self._render_text(score_str, (self.WIDTH / 2, 10), self.font_medium, self.COLOR_TEXT, align="center")

        # Game Over / Stage Clear messages
        if self.stage_complete_frames > 0:
            if self.stage > self.MAX_STAGES:
                 self._render_text("YOU WIN!", (self.WIDTH/2, self.HEIGHT/2 - 40), self.font_large, (255,215,0))
            else:
                 self._render_text(f"STAGE {self.stage} CLEAR!", (self.WIDTH/2, self.HEIGHT/2 - 40), self.font_large, self.COLOR_TEXT)
        elif self.game_over and not self.win_game:
            self._render_text("TIME'S UP!", (self.WIDTH/2, self.HEIGHT/2 - 40), self.font_large, self.COLOR_PLAYER)

    def _render_text(self, text, pos, font, color, align="center", shadow=True):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        if align == "center":
            text_rect.center = pos
        elif align == "left":
            text_rect.topleft = pos
        elif align == "right":
            text_rect.topright = pos
        
        if shadow:
            shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        
        self.screen.blit(text_surf, text_rect)

    def _create_particles(self, pos, count, color, min_speed, max_speed, lifetime=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'size': self.np_random.uniform(2, 5),
                'color': color,
                'lifetime': self.np_random.integers(lifetime // 2, lifetime)
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify the implementation.
        """
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2], f"Action space nvec is {self.action_space.nvec.tolist()}"
        
        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8, f"Obs dtype is {test_obs.dtype}"
        
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
        assert not trunc, "Truncated should always be False"
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


# Example usage:
if __name__ == '__main__':
    env = GameEnv()
    env.reset(seed=42)
    
    # --- Manual Play ---
    # This loop allows for manual play to test the game feel and visuals.
    # It simulates a simple agent that holds the 'up' key.
    # To play yourself, you would need to map keyboard inputs to actions.
    
    terminated = False
    total_reward = 0
    
    # Pygame setup for visualization
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Snail Racer")
    clock = pygame.time.Clock()
    
    # Action state
    action = env.action_space.sample()
    action[0] = 0 # No-op initially
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Simple Keyboard Controls for Testing ---
        keys = pygame.key.get_pressed()
        action = [0, 0, 0] # Reset action
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            # In a real scenario, you'd reset the env here
            # For this demo, we'll just stop after one episode
            pygame.time.wait(3000) # Pause for 3 seconds to see the final screen
            running = False
        
        clock.tick(GameEnv.FPS)

    env.close()