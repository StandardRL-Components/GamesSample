import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Snail:
    def __init__(self, color, pos, name):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(0, 0)
        self.color = color
        self.shell_color = tuple(max(0, c - 40) for c in color)
        self.dark_color = tuple(max(0, c - 80) for c in color)
        self.name = name
        self.on_ground = True
        self.boost_cooldown = 0
        self.rank = 0
        self.finished = False
        self.finish_time = float('inf')
        self.collided_obstacle = False

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑ to accelerate, ↓ to brake, ←→ to steer. Press space for a speed boost."
    )

    game_description = (
        "Fast-paced arcade snail racer. Steer your snail along the track, use boosts, and outpace your opponents to the finish line."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.TRACK_LENGTH = 8000
        self.TRACK_THICKNESS = 120
        self.FINISH_LINE_X = self.TRACK_LENGTH - 200

        # Game constants
        self.FPS = 30
        self.MAX_STEPS = 60 * self.FPS  # 60 seconds

        # Physics constants
        self.GRAVITY = 0.4
        self.ACCELERATION = 0.25
        self.BRAKE_FORCE = 0.4
        self.STEER_FORCE = 0.5
        self.FRICTION = 0.98
        self.BOOST_FORCE = 6.0
        self.BOOST_COOLDOWN = 90  # 3 seconds
        self.MAX_SPEED_X = 10
        self.COLLISION_DAMPING = 0.5

        # Colors
        self.COLOR_BG = (135, 206, 235) # Sky Blue
        self.COLOR_BG_DARK = (100, 160, 190)
        self.COLOR_TRACK_TOP = (124, 252, 0) # Lawn Green
        self.COLOR_TRACK_FILL = (34, 139, 34) # Forest Green
        self.COLOR_OBSTACLE = (139, 69, 19) # Saddle Brown
        self.COLOR_OBSTACLE_DARK = (90, 45, 12)
        self.COLOR_PLAYER = (0, 120, 255) # Bright Blue
        self.COLOR_AI1 = (255, 165, 0) # Orange
        self.COLOR_AI2 = (255, 215, 0) # Gold
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (50, 50, 50)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_BLACK = (0, 0, 0)
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Initialize state variables
        self.player = None
        self.ais = []
        self.all_snails = []
        self.track_points = []
        self.obstacles = []
        self.particles = []
        self.camera_x = 0.0
        self.np_random = None
        
    def _generate_track(self):
        self.track_points = []
        y = self.SCREEN_HEIGHT * 0.7
        for x in range(self.TRACK_LENGTH + self.SCREEN_WIDTH):
            y += self.np_random.uniform(-1.5, 1.5)
            y += math.sin(x / 200) * 1.0
            y += math.sin(x / 80) * 0.5
            y = np.clip(y, self.SCREEN_HEIGHT * 0.4, self.SCREEN_HEIGHT * 0.9)
            self.track_points.append(y)

    def _generate_obstacles(self):
        self.obstacles = []
        for x in range(1000, self.FINISH_LINE_X - 500, 400):
            if self.np_random.random() > 0.4:
                track_y = self._get_track_y(x)
                obstacle_y = track_y + self.np_random.uniform(20, self.TRACK_THICKNESS - 20)
                radius = self.np_random.uniform(10, 25)
                self.obstacles.append({'pos': pygame.Vector2(x, obstacle_y), 'radius': radius})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.MAX_STEPS
        
        self._generate_track()
        self._generate_obstacles()

        player_start_pos = (100, self._get_track_y(100))
        self.player = Snail(self.COLOR_PLAYER, player_start_pos, "Player")
        
        self.ais = [
            Snail(self.COLOR_AI1, (80, self._get_track_y(80)), "AI 1"),
            Snail(self.COLOR_AI2, (60, self._get_track_y(60)), "AI 2")
        ]
        
        self.all_snails = [self.player] + self.ais
        for snail in self.all_snails:
            snail.finished = False
            snail.finish_time = float('inf')

        self.last_ranks = self._get_ranks()
        
        self.camera_x = 0
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def _update_player(self, movement, space_held):
        snail = self.player
        if snail.finished:
            return

        # Handle controls
        if movement == 1:  # Up
            snail.vel.x += self.ACCELERATION
        elif movement == 2:  # Down
            snail.vel.x -= self.BRAKE_FORCE
        elif movement == 3:  # Left
            snail.vel.y -= self.STEER_FORCE
        elif movement == 4:  # Right
            snail.vel.y += self.STEER_FORCE

        # Handle boost
        if space_held and snail.boost_cooldown == 0:
            snail.vel.x += self.BOOST_FORCE
            snail.boost_cooldown = self.BOOST_COOLDOWN
            for _ in range(30):
                angle = self.np_random.uniform(math.pi * 0.9, math.pi * 1.1)
                speed = self.np_random.uniform(4, 8)
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                self.particles.append({'pos': pygame.Vector2(snail.pos), 'vel': vel, 'life': 20, 'color': self.COLOR_WHITE, 'radius': self.np_random.uniform(1, 4)})

        # Update boost cooldown
        if snail.boost_cooldown > 0:
            snail.boost_cooldown -= 1
        
        self._update_snail_physics(snail)

    def _update_ais(self):
        for ai in self.ais:
            if ai.finished:
                continue
            
            # Simple AI: try to stay in the middle of the track and maintain speed
            track_y_top = self._get_track_y(ai.pos.x)
            track_y_bottom = track_y_top + self.TRACK_THICKNESS
            target_y = (track_y_top + track_y_bottom) / 2
            
            if ai.pos.y < target_y - 5:
                ai.vel.y += self.STEER_FORCE * 0.5 * self.np_random.uniform(0.8, 1.2)
            elif ai.pos.y > target_y + 5:
                ai.vel.y -= self.STEER_FORCE * 0.5 * self.np_random.uniform(0.8, 1.2)
            
            target_speed = self.MAX_SPEED_X * self.np_random.uniform(0.75, 0.9)
            if ai.vel.x < target_speed:
                ai.vel.x += self.ACCELERATION * 0.8
            
            self._update_snail_physics(ai)

    def _update_snail_physics(self, snail):
        # Apply friction and gravity
        snail.vel.x *= self.FRICTION
        snail.vel.x = max(0, snail.vel.x)
        if not snail.on_ground:
            snail.vel.y += self.GRAVITY
        
        # Clamp speed
        snail.vel.x = min(snail.vel.x, self.MAX_SPEED_X)
        snail.vel.y = np.clip(snail.vel.y, -self.MAX_SPEED_X / 2, self.MAX_SPEED_X / 2)
        
        # Update position
        snail.pos += snail.vel
        
        # Track collision
        track_y_top = self._get_track_y(snail.pos.x)
        track_y_bottom = track_y_top + self.TRACK_THICKNESS
        
        snail.on_ground = False
        if track_y_top <= snail.pos.y <= track_y_bottom:
            snail.on_ground = True
            snail.vel.y *= 0.8 # Dampen vertical movement
        elif snail.pos.y < track_y_top:
            # In the air, but still above the track
            pass
        else: # Below the track
            snail.on_ground = False
            if snail is self.player:
                self.game_over = True
                self.score -= 100 # Terminal penalty
        
        # Obstacle collision
        snail.collided_obstacle = False
        for obs in self.obstacles:
            if snail.pos.distance_to(obs['pos']) < obs['radius'] + 15: # 15 is snail radius
                snail.vel.x *= self.COLLISION_DAMPING
                snail.collided_obstacle = True
                for _ in range(15):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(1, 4)
                    vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                    self.particles.append({'pos': pygame.Vector2(snail.pos), 'vel': vel, 'life': 15, 'color': self.COLOR_OBSTACLE, 'radius': self.np_random.uniform(1, 3)})
                break
        
        # Finish line
        if not snail.finished and snail.pos.x >= self.FINISH_LINE_X:
            snail.finished = True
            snail.finish_time = self.steps

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _update_camera(self):
        lead_x = 0
        if self.all_snails:
            lead_x = max(s.pos.x for s in self.all_snails)
        
        target_camera_x = lead_x - self.SCREEN_WIDTH * 0.3
        self.camera_x = self.camera_x * 0.9 + target_camera_x * 0.1

    def _get_ranks(self):
        # Sort snails by finished status, then finish time, then x position
        sorted_snails = sorted(self.all_snails, key=lambda s: (not s.finished, s.finish_time, -s.pos.x))
        ranks = {snail.name: i for i, snail in enumerate(sorted_snails)}
        for snail in self.all_snails:
            snail.rank = ranks[snail.name]
        return ranks

    def _calculate_reward(self, movement):
        reward = 0
        
        # Continuous rewards
        reward += self.player.vel.x * 0.01  # Reward for forward speed
        if movement == 2: # Braking
            reward -= 0.02
        
        # Event-based rewards
        if self.player.collided_obstacle:
            reward -= 0.5
            
        current_ranks = self._get_ranks()
        player_old_rank = self.last_ranks.get(self.player.name, len(self.all_snails) - 1)
        player_new_rank = current_ranks.get(self.player.name, len(self.all_snails) - 1)
        
        if player_new_rank < player_old_rank:
            reward += 5 # Overtook an AI
        elif player_new_rank > player_old_rank:
            reward -= 1 # Was overtaken
            
        self.last_ranks = current_ranks
        return reward

    def _check_termination(self):
        self.time_left -= 1
        if self.time_left <= 0:
            self.game_over = True
            self.score -= 50 # Time out penalty

        all_finished = all(s.finished for s in self.all_snails)
        if self.player.finished or all_finished:
            self.game_over = True

        if self.game_over:
            # Assign terminal rewards only once
            if not self.player.finished and self.player.pos.y > self._get_track_y(self.player.pos.x) + self.TRACK_THICKNESS:
                 pass # Already penalized for falling
            else:
                ranks = self._get_ranks()
                player_rank = ranks.get(self.player.name)
                if player_rank == 0:
                    self.score += 100 # 1st place
                elif player_rank == 1:
                    self.score += 50 # 2nd place
                elif player_rank == 2:
                    self.score += 25 # 3rd place
        
        return self.game_over

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_held = action[1] == 1
        
        self._update_player(movement, space_held)
        self._update_ais()
        self._update_particles()
        self._update_camera()

        reward = self._calculate_reward(movement)
        self.score += reward
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False, # We use terminated for both game over and timeout
            self._get_info()
        )
    
    def _get_track_y(self, x):
        x = int(x)
        if 0 <= x < len(self.track_points):
            return self.track_points[x]
        elif x < 0:
            return self.track_points[0]
        else:
            return self.track_points[-1]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Parallax background
        for i in range(3):
            offset = (self.camera_x * (0.1 * (i + 1))) % self.SCREEN_WIDTH
            pygame.draw.circle(self.screen, self.COLOR_BG_DARK, (100 - offset, 100 + i*30), 40 - i*10, 2)
            pygame.draw.circle(self.screen, self.COLOR_BG_DARK, (400 - offset, 80 + i*20), 50 - i*10, 2)
            pygame.draw.circle(self.screen, self.COLOR_BG_DARK, (700 - offset, 120 + i*40), 60 - i*10, 2)

        # Track
        cam_x_int = int(self.camera_x)
        for x in range(self.SCREEN_WIDTH + 1):
            world_x = cam_x_int + x
            if 0 <= world_x < len(self.track_points):
                y_top = self.track_points[world_x]
                pygame.draw.line(self.screen, self.COLOR_TRACK_FILL, (x, y_top), (x, y_top + self.TRACK_THICKNESS), 1)
                pygame.gfxdraw.pixel(self.screen, x, int(y_top), self.COLOR_TRACK_TOP)

        # Finish line
        finish_screen_x = self.FINISH_LINE_X - cam_x_int
        if 0 < finish_screen_x < self.SCREEN_WIDTH:
            for y in range(0, self.SCREEN_HEIGHT, 10):
                color = self.COLOR_WHITE if (y // 10) % 2 == 0 else self.COLOR_BLACK
                pygame.draw.rect(self.screen, color, (finish_screen_x, y, 10, 10))
            pygame.draw.line(self.screen, self.COLOR_BLACK, (finish_screen_x, 0), (finish_screen_x, self.SCREEN_HEIGHT), 2)

        # Obstacles
        for obs in self.obstacles:
            screen_pos = obs['pos'] - pygame.Vector2(self.camera_x, 0)
            if -obs['radius'] < screen_pos.x < self.SCREEN_WIDTH + obs['radius']:
                pygame.draw.circle(self.screen, self.COLOR_OBSTACLE_DARK, screen_pos, obs['radius'])
                pygame.draw.circle(self.screen, self.COLOR_OBSTACLE, screen_pos, obs['radius']-3)

        # Particles
        for p in self.particles:
            screen_pos = p['pos'] - pygame.Vector2(self.camera_x, 0)
            alpha = max(0, min(255, int(255 * (p['life'] / 20.0))))
            color = (*p['color'], alpha)
            s = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(s, (int(screen_pos.x - p['radius']), int(screen_pos.y - p['radius'])))

        # Snails
        for snail in sorted(self.all_snails, key=lambda s: s.pos.y):
             self._draw_snail(snail)
        
        # Speed lines
        if self.player.vel.x > self.MAX_SPEED_X * 0.8:
            for _ in range(5):
                start_x = self.np_random.uniform(-self.SCREEN_WIDTH * 0.1, self.SCREEN_WIDTH * 1.1)
                start_y = self.np_random.uniform(0, self.SCREEN_HEIGHT)
                length = self.player.vel.x * self.np_random.uniform(1.5, 3.0)
                alpha = int(100 * (self.player.vel.x / self.MAX_SPEED_X))
                try:
                    pygame.draw.line(self.screen, (*self.COLOR_WHITE, alpha), (start_x, start_y), (start_x - length, start_y), 2)
                except TypeError: # Color may not support alpha
                    pygame.draw.line(self.screen, self.COLOR_WHITE, (start_x, start_y), (start_x - length, start_y), 2)

    def _draw_snail(self, snail):
        screen_pos = snail.pos - pygame.Vector2(self.camera_x, 0)
        
        # Body
        body_rect = pygame.Rect(screen_pos.x - 15, screen_pos.y - 5, 30, 10)
        pygame.draw.ellipse(self.screen, snail.dark_color, body_rect.inflate(2, 2))
        pygame.draw.ellipse(self.screen, snail.color, body_rect)
        
        # Head
        head_pos = (screen_pos.x + 12, screen_pos.y - 6)
        pygame.draw.circle(self.screen, snail.dark_color, head_pos, 7)
        pygame.draw.circle(self.screen, snail.color, head_pos, 6)
        
        # Eye
        eye_pos = (head_pos[0] + 2, head_pos[1] - 1)
        pygame.draw.circle(self.screen, self.COLOR_WHITE, eye_pos, 3)
        pygame.draw.circle(self.screen, self.COLOR_BLACK, eye_pos, 1)

        # Shell
        shell_pos = (screen_pos.x - 5, screen_pos.y - 15)
        pygame.draw.circle(self.screen, snail.dark_color, shell_pos, 12)
        pygame.draw.circle(self.screen, snail.shell_color, shell_pos, 11)
        # Spiral
        for i in range(5):
            angle = i * math.pi / 2 + (self.steps * 0.05)
            r = 10 - i * 2
            p1_x = shell_pos[0] + r * math.cos(angle)
            p1_y = shell_pos[1] + r * math.sin(angle)
            p2_x = shell_pos[0] + (r-2) * math.cos(angle + math.pi/2)
            p2_y = shell_pos[1] + (r-2) * math.sin(angle + math.pi/2)
            pygame.draw.line(self.screen, snail.dark_color, (p1_x, p1_y), (p2_x, p2_y), 2)

        # Rank text
        rank_str = {0: "1st", 1: "2nd", 2: "3rd"}.get(snail.rank, "")
        if snail.finished:
            rank_str += " ✓"
        
        text = self.font_small.render(rank_str, True, self.COLOR_WHITE)
        text_shadow = self.font_small.render(rank_str, True, self.COLOR_TEXT_SHADOW)
        text_pos = (screen_pos.x - text.get_width() / 2, screen_pos.y - 50)
        self.screen.blit(text_shadow, (text_pos[0] + 1, text_pos[1] + 1))
        self.screen.blit(text, text_pos)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_WHITE)
        score_shadow = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(score_shadow, (11, 11))
        self.screen.blit(score_text, (10, 10))
        
        # Time
        time_sec = self.time_left // self.FPS
        time_text = self.font_large.render(f"Time: {time_sec}", True, self.COLOR_WHITE)
        time_shadow = self.font_large.render(f"Time: {time_sec}", True, self.COLOR_TEXT_SHADOW)
        time_pos = (self.SCREEN_WIDTH - time_text.get_width() - 10, 10)
        self.screen.blit(time_shadow, (time_pos[0] + 1, time_pos[1] + 1))
        self.screen.blit(time_text, time_pos)

        # Progress bar
        progress_y = self.SCREEN_HEIGHT - 20
        pygame.draw.rect(self.screen, self.COLOR_BLACK, (10, progress_y, self.SCREEN_WIDTH - 20, 10))
        for i, snail in enumerate(self.all_snails):
            progress = snail.pos.x / self.FINISH_LINE_X
            bar_width = self.SCREEN_WIDTH - 20
            marker_x = 10 + progress * bar_width
            pygame.draw.circle(self.screen, snail.color, (marker_x, progress_y + 5), 8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": (self.player.pos.x, self.player.pos.y),
            "player_vel": (self.player.vel.x, self.player.vel.y),
            "time_left": self.time_left,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # Unset the dummy video driver for manual play
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    # Pygame setup for manual play
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Snail Racer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    action = np.array([0, 0, 0])

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space, shift])
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            # Wait for a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    env.close()