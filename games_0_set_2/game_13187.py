import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import random
from gymnasium.spaces import MultiDiscrete
import numpy as np
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a surreal racing game.

    The player races through a morphing dream-reality track, matching tiles
    to alter the course and upgrade their vehicle. The core gameplay loop involves
    observing the track ahead, matching tiles to influence track changes,
    controlling the vehicle, collecting tokens, and making narrative choices.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Space button (0=released, 1=held) for tile matching
    - actions[2]: Shift button (0=released, 1=held) reserved

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Race through a surreal, morphing dreamscape. Match tiles on the track to alter the course and avoid obstacles."
    user_guide = "Controls: ↑ to accelerate, ↓ to brake, ←→ to steer. Press space to match tiles on the track."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 5000

    # Colors
    COLOR_BG_DREAM_1 = (40, 0, 60)
    COLOR_BG_DREAM_2 = (10, 0, 20)
    COLOR_REALITY_1 = (20, 40, 60)
    COLOR_REALITY_2 = (5, 10, 20)

    COLOR_TRACK_REALITY = (60, 120, 180)
    COLOR_TRACK_DREAM = (180, 60, 220)
    COLOR_TRACK_LINE = (200, 220, 255)

    COLOR_TILE_REALITY = (100, 180, 255, 180)
    COLOR_TILE_DREAM = (255, 100, 220, 180)

    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (0, 255, 255)
    COLOR_TOKEN = (255, 223, 0)
    COLOR_OBSTACLE = (255, 50, 50)

    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_SHADOW = (0, 0, 0)

    # Player Physics
    ACCELERATION = 0.2
    BRAKING = 0.4
    FRICTION = 0.04
    MAX_SPEED = 5.0
    STEER_SPEED = 4.0
    CENTRIFUGAL_FORCE = 0.03

    # Track
    TRACK_WIDTH = 180
    SEGMENT_LENGTH = 20
    NUM_SEGMENTS = 250
    CHECKPOINT_INTERVAL = 50

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)

        self.bg_hue = 0
        self.render_mode = render_mode

        # Initialize state variables
        self.player_progress = 0.0
        self.player_vel = 0.0
        self.player_x_offset = 0.0
        self.track_points = []
        self.tiles = []
        self.tokens = []
        self.obstacles = []
        self.particles = []
        self.checkpoints = set()
        self.tile_morph_speed = 1.0
        self.match_cooldown = 0
        self.steps = 0
        self.score = 0
        self.token_count = 0
        self.game_over = False  # For crash state
        self.terminated = False
        self.truncated = False

        # self.reset() # Not strictly necessary, but good practice

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.token_count = 0
        self.game_over = False
        self.terminated = False
        self.truncated = False

        self.player_progress = 0.0
        self.player_vel = 0.0
        self.player_x_offset = 0.0

        self.particles = []
        self.tile_morph_speed = 1.0
        self.match_cooldown = 0

        self._generate_track()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.terminated or self.truncated:
            return self._get_observation(), 0.0, self.terminated, self.truncated, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = 0.0

        # --- Update Game Logic ---

        # 1. Handle Player Input
        self._handle_input(movement)

        # 2. Update Player Position
        self.player_vel = max(0, self.player_vel - self.FRICTION)
        self.player_progress += self.player_vel

        # Apply centrifugal force
        _, _, curve = self._get_track_segment(self.player_progress)
        self.player_x_offset -= self.player_vel * curve * self.CENTRIFUGAL_FORCE

        # Keep player on track
        self.player_x_offset = np.clip(self.player_x_offset, -self.TRACK_WIDTH / 2, self.TRACK_WIDTH / 2)

        if self.player_vel > 0.1:
            reward += 0.1  # Reward for forward movement

        # 3. Update Game World
        self._update_particles()
        self._update_tiles()
        if self.match_cooldown > 0:
            self.match_cooldown -= 1

        # 4. Handle Interactions
        if space_held and self.match_cooldown == 0:
            reward += self._handle_tile_match()

        reward += self._check_collisions()

        # 5. Difficulty Scaling
        if self.steps > 0 and self.steps % 500 == 0:
            self.tile_morph_speed += 0.05

        # 6. Check Termination and Truncation
        terminated = self.game_over or self.player_progress >= self.NUM_SEGMENTS
        truncated = self.steps >= self.MAX_STEPS

        if terminated:
            if self.player_progress >= self.NUM_SEGMENTS:
                reward += 100  # Victory reward
            else:  # Crash
                reward -= 100  # Crash penalty

        self.terminated = terminated
        self.truncated = truncated

        return (
            self._get_observation(),
            reward,
            self.terminated,
            self.truncated,
            self._get_info(),
        )

    def _handle_input(self, movement):
        # movement: 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1:  # Accelerate
            self.player_vel += self.ACCELERATION
        elif movement == 2:  # Brake
            self.player_vel -= self.BRAKING
        if movement == 3:  # Steer Left
            self.player_x_offset -= self.STEER_SPEED
        elif movement == 4:  # Steer Right
            self.player_x_offset += self.STEER_SPEED

        self.player_vel = np.clip(self.player_vel, 0, self.MAX_SPEED)

    def _generate_track(self):
        self.track_points = []
        self.tiles = []
        self.tokens = []
        self.obstacles = []
        self.checkpoints = set(range(self.CHECKPOINT_INTERVAL, self.NUM_SEGMENTS, self.CHECKPOINT_INTERVAL))

        x, y, curve = 0.0, 0.0, 0.0
        for i in range(self.NUM_SEGMENTS + 50):  # Generate extra for smooth rendering
            self.track_points.append({'x': x, 'y': y, 'curve': curve})
            x += curve
            y += self.SEGMENT_LENGTH

            if self.np_random.random() < 0.1:
                curve += self.np_random.uniform(-0.05, 0.05)
                curve = np.clip(curve, -0.2, 0.2)

            # Generate tiles, tokens, obstacles for the main track part
            if i < self.NUM_SEGMENTS and i > 5:
                # Tiles (3 lanes)
                for offset in [-self.TRACK_WIDTH / 3, 0, self.TRACK_WIDTH / 3]:
                    state = self.np_random.integers(0, 2)
                    self.tiles.append(
                        {'progress': i, 'offset': offset, 'state': state, 'morph_timer': self.np_random.uniform(0, 100)})

                # Tokens
                if self.np_random.random() < 0.05:
                    offset = self.np_random.uniform(-self.TRACK_WIDTH / 2 + 20, self.TRACK_WIDTH / 2 - 20)
                    self.tokens.append({'progress': i, 'offset': offset, 'collected': False})

                # Obstacles
                elif self.np_random.random() < 0.03:
                    offset = self.np_random.uniform(-self.TRACK_WIDTH / 2 + 20, self.TRACK_WIDTH / 2 - 20)
                    self.obstacles.append({'progress': i, 'offset': offset, 'hit': False})

    def _get_track_segment(self, progress):
        idx = int(progress) % len(self.track_points)
        p1 = self.track_points[idx]
        p2 = self.track_points[(idx + 1) % len(self.track_points)]

        interp = progress - idx
        x = p1['x'] + (p2['x'] - p1['x']) * interp
        y = p1['y'] + (p2['y'] - p1['y']) * interp
        curve = p1['curve'] + (p2['curve'] - p1['curve']) * interp
        return x, y, curve

    def _project(self, world_x, world_y, world_z, cam_x, cam_y, cam_z):
        dx, dy, dz = world_x - cam_x, world_y - cam_y, world_z - cam_z
        if dz <= 0: return None

        scale = self.HEIGHT / dz
        screen_x = self.WIDTH / 2 + scale * dx
        screen_y = self.HEIGHT / 2 - scale * dy
        return int(screen_x), int(screen_y), scale

    def _handle_tile_match(self):
        player_tile_idx = -1
        min_dist_sq = float('inf')

        # Find the tile the player is on
        for i, tile in enumerate(self.tiles):
            if abs(tile['progress'] - self.player_progress) < 0.5:
                dist_sq = (tile['offset'] - self.player_x_offset) ** 2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    player_tile_idx = i

        if player_tile_idx == -1: return 0

        player_tile = self.tiles[player_tile_idx]

        # Find adjacent tiles at the same progress level
        adjacent_tiles = []
        for i, tile in enumerate(self.tiles):
            if i != player_tile_idx and abs(tile['progress'] - player_tile['progress']) < 0.5:
                if abs(tile['offset'] - player_tile['offset']) < self.TRACK_WIDTH / 2:
                    adjacent_tiles.append(tile)

        # Check for a match
        for adj_tile in adjacent_tiles:
            if adj_tile['state'] == player_tile['state']:
                # Match found! Flip states
                player_tile['state'] = 1 - player_tile['state']
                adj_tile['state'] = 1 - adj_tile['state']

                # Add visual effect
                self._create_particles(self.WIDTH / 2, self.HEIGHT - 50, 20, self.COLOR_TOKEN, 'burst')
                self.match_cooldown = 15  # 0.5 seconds
                return 5.0  # Reward for matching

        return 0.0

    def _check_collisions(self):
        reward = 0

        # Tokens
        for token in self.tokens:
            if not token['collected'] and abs(token['progress'] - self.player_progress) < 0.5:
                if abs(token['offset'] - self.player_x_offset) < 20:
                    token['collected'] = True
                    self.score += 10
                    self.token_count += 1
                    reward += 0.2
                    self._create_particles(self.WIDTH / 2, self.HEIGHT - 50, 10, self.COLOR_TOKEN, 'trail')

        # Obstacles
        for obstacle in self.obstacles:
            if not obstacle['hit'] and abs(obstacle['progress'] - self.player_progress) < 0.5:
                if abs(obstacle['offset'] - self.player_x_offset) < 20:
                    obstacle['hit'] = True
                    self.player_vel *= 0.1
                    self.game_over = True  # Set crash flag
                    reward -= 0.5
                    self._create_particles(self.WIDTH / 2, self.HEIGHT - 50, 30, self.COLOR_OBSTACLE, 'burst')

        # Checkpoints
        current_segment = int(self.player_progress)
        if current_segment in self.checkpoints:
            self.checkpoints.remove(current_segment)
            self.score += 50
            reward += 10.0

        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['life'] -= 1
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += p.get('gravity', 0)
            p['size'] = max(0, p['size'] - 0.1)

    def _create_particles(self, x, y, count, color, p_type):
        for _ in range(count):
            if p_type == 'trail':
                self.particles.append({
                    'x': x + self.np_random.uniform(-5, 5), 'y': y + self.np_random.uniform(-5, 5),
                    'vx': self.np_random.uniform(-1, 1), 'vy': self.np_random.uniform(-2, 0),
                    'size': self.np_random.uniform(2, 5), 'life': 20, 'color': color
                })
            elif p_type == 'burst':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 5)
                self.particles.append({
                    'x': x, 'y': y,
                    'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                    'size': self.np_random.uniform(4, 8), 'life': 30, 'color': color
                })

    def _update_tiles(self):
        for tile in self.tiles:
            tile['morph_timer'] -= self.tile_morph_speed
            if tile['morph_timer'] <= 0:
                tile['state'] = 1 - tile['state']
                tile['morph_timer'] = self.np_random.uniform(100, 200)

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "tokens": self.token_count,
            "progress": self.player_progress / self.NUM_SEGMENTS if self.NUM_SEGMENTS > 0 else 1.0,
            "speed": self.player_vel
        }

    def _render_game(self):
        # --- Background ---
        self.bg_hue = (self.bg_hue + 0.1) % 360
        color1 = pygame.Color(0)
        color1.hsva = (self.bg_hue, 80, 20, 100)
        color2 = pygame.Color(0)
        color2.hsva = ((self.bg_hue + 20) % 360, 90, 40, 100)

        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = color1.lerp(color2, interp)
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        # --- Camera and Track Rendering ---
        cam_progress = self.player_progress
        cam_x, cam_y, _ = self._get_track_segment(cam_progress)
        cam_z = -self.HEIGHT
        cam_x += self.player_x_offset

        # Render from horizon down to player
        last_p = None
        for i in range(int(cam_progress) + 50, int(cam_progress), -1):
            idx = i % len(self.track_points)
            track_seg = self.track_points[idx]

            p = self._project(track_seg['x'], track_seg['y'] + 100, 0, cam_x, cam_y, cam_z)
            if p is None: continue

            if last_p is not None:
                w1 = last_p[2] * self.TRACK_WIDTH
                w2 = p[2] * self.TRACK_WIDTH

                # Track ground
                pygame.draw.polygon(self.screen, self.COLOR_TRACK_REALITY, [
                    (last_p[0] - w1, last_p[1]), (last_p[0] + w1, last_p[1]),
                    (p[0] + w2, p[1]), (p[0] - w2, p[1])
                ])
                # Track lines
                pygame.draw.line(self.screen, self.COLOR_TRACK_LINE, (last_p[0] - w1, last_p[1]), (p[0] - w2, p[1]), 2)
                pygame.draw.line(self.screen, self.COLOR_TRACK_LINE, (last_p[0] + w1, last_p[1]), (p[0] + w2, p[1]), 2)

            last_p = p

        # --- Render Game Objects (Tiles, Tokens, Obstacles) ---
        for i in range(int(cam_progress) + 50, int(cam_progress), -1):
            # Tiles
            for tile in [t for t in self.tiles if int(t['progress']) == i]:
                self._render_object(tile, cam_x, cam_y, cam_z, self._render_tile)
            # Tokens
            for token in [t for t in self.tokens if int(t['progress']) == i and not t['collected']]:
                self._render_object(token, cam_x, cam_y, cam_z, self._render_token)
            # Obstacles
            for obstacle in [o for o in self.obstacles if int(o['progress']) == i and not o['hit']]:
                self._render_object(obstacle, cam_x, cam_y, cam_z, self._render_obstacle)

        # --- Player Trail ---
        if self.player_vel > 1.0:
            self._create_particles(self.WIDTH / 2, self.HEIGHT - 40, 1, self.COLOR_PLAYER_GLOW, 'trail')

        # --- Particles ---
        for p in self.particles:
            size = int(p['size'])
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), size)

        # --- Player Vehicle ---
        player_y = self.HEIGHT - 40
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, self.WIDTH // 2, player_y, 14, (*self.COLOR_PLAYER_GLOW, 50))
        pygame.gfxdraw.filled_circle(self.screen, self.WIDTH // 2, player_y, 10, (*self.COLOR_PLAYER_GLOW, 100))
        # Body
        player_points = [
            (self.WIDTH / 2, player_y - 10),
            (self.WIDTH / 2 - 8, player_y + 8),
            (self.WIDTH / 2 + 8, player_y + 8)
        ]
        pygame.gfxdraw.aapolygon(self.screen, player_points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, player_points, self.COLOR_PLAYER)

        # --- UI ---
        self._render_ui()

    def _render_object(self, obj, cam_x, cam_y, cam_z, render_func):
        obj_progress = obj['progress']
        obj_offset = obj['offset']

        obj_track_x, obj_track_y, _ = self._get_track_segment(obj_progress)

        p = self._project(obj_track_x + obj_offset, obj_track_y, 0, cam_x, cam_y, cam_z)
        if p is not None:
            render_func(obj, p[0], p[1], p[2])

    def _render_tile(self, tile, x, y, scale):
        color = self.COLOR_TILE_DREAM if tile['state'] == 1 else self.COLOR_TILE_REALITY
        width = scale * (self.TRACK_WIDTH / 3.5)
        height = scale * self.SEGMENT_LENGTH

        if height < 1 or width < 1: return

        # Pulsing alpha for dream tiles
        alpha_color = list(color)
        if tile['state'] == 1:
            alpha_color[3] = int(128 + 64 * math.sin(pygame.time.get_ticks() / 200))

        rect_surf = pygame.Surface((width, height), pygame.SRCALPHA)
        rect_surf.fill(alpha_color)
        self.screen.blit(rect_surf, (x - width / 2, y - height / 2))

    def _render_token(self, token, x, y, scale):
        size = int(max(5, scale * 15))
        angle = (pygame.time.get_ticks() / 10) % 360

        outer_pts = []
        inner_pts = []
        for i in range(5):
            rad = math.radians(angle + i * 72)
            outer_pts.append((x + size * math.cos(rad), y - size * math.sin(rad)))
            rad = math.radians(angle + 36 + i * 72)
            inner_pts.append((x + size / 2 * math.cos(rad), y - size / 2 * math.sin(rad)))

        all_pts = []
        for i in range(5):
            all_pts.append(outer_pts[i])
            all_pts.append(inner_pts[i])

        pygame.gfxdraw.aapolygon(self.screen, all_pts, self.COLOR_TOKEN)
        pygame.gfxdraw.filled_polygon(self.screen, all_pts, self.COLOR_TOKEN)

    def _render_obstacle(self, obstacle, x, y, scale):
        size = int(max(4, scale * 20))
        points = [
            (x, y - size),
            (x + size, y),
            (x, y + size),
            (x - size, y),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_OBSTACLE)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_OBSTACLE)

    def _render_ui(self):
        def draw_text(text, x, y, font, color, shadow_color):
            text_surf = font.render(text, True, shadow_color)
            self.screen.blit(text_surf, (x + 2, y + 2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, (x, y))

        score_text = f"SCORE: {self.score}"
        draw_text(score_text, 10, 10, self.font_small, self.COLOR_UI_TEXT, self.COLOR_UI_SHADOW)

        tokens_text = f"TOKENS: {self.token_count}"
        draw_text(tokens_text, 10, 30, self.font_small, self.COLOR_TOKEN, self.COLOR_UI_SHADOW)

        speed_text = f"SPEED: {self.player_vel:.1f}"
        draw_text(speed_text, self.WIDTH - 150, 10, self.font_small, self.COLOR_UI_TEXT, self.COLOR_UI_SHADOW)

        progress = int((self.player_progress / self.NUM_SEGMENTS * 100) if self.NUM_SEGMENTS > 0 else 0)
        progress_text = f"PROGRESS: {progress}%"
        draw_text(progress_text, self.WIDTH - 150, 30, self.font_small, self.COLOR_UI_TEXT, self.COLOR_UI_SHADOW)

        if self.terminated or self.truncated:
            if self.player_progress >= self.NUM_SEGMENTS:
                msg = "FINISH"
            elif self.game_over:  # This is only true on crash
                msg = "CRASHED"
            else:  # Truncated
                msg = "TIME UP"

            shadow_surf = self.font_large.render(msg, True, self.COLOR_UI_SHADOW)
            shadow_rect = shadow_surf.get_rect(center=(self.WIDTH / 2 + 2, self.HEIGHT / 2 - 13))
            self.screen.blit(shadow_surf, shadow_rect)

            text_surf = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 15))
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # It requires pygame to be installed with display support.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    
    env = GameEnv(render_mode="human_playable")
    obs, info = env.reset()
    done = False

    # --- Human Player Controls ---
    # ARROWS: move
    # SPACE: match tiles
    # R: reset
    # Q: quit

    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Dream Racer")
    clock = pygame.time.Clock()

    action = [0, 0, 0]  # no-op, no-space, no-shift

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                if event.key == pygame.K_q:
                    done = True

        keys = pygame.key.get_pressed()

        # Movement
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action[0] = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action[0] = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action[0] = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action[0] = 4
        else:
            action[0] = 0

        # Space
        action[1] = 1 if keys[pygame.K_SPACE] else 0

        # Shift
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over. Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(2000)  # Pause for 2 seconds before reset
            obs, info = env.reset()

        clock.tick(GameEnv.FPS)

    env.close()