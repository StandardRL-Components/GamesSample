import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls an antibody to destroy bacteria.

    The player aims the antibody and launches it using a magnetic field. The goal is
    to clear all bacteria from the level without colliding with them during the aiming phase.
    The game features time dilation during aiming for strategic precision.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # --- Game Metadata ---
    game_description = (
        "Control an antibody to clear the arena of moving bacteria. Aim and launch your antibody to destroy them, but avoid collisions while aiming."
    )
    user_guide = (
        "Use ← and → arrow keys to aim the antibody. Press space to launch."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    CENTER = pygame.math.Vector2(WIDTH / 2, HEIGHT / 2)
    ARENA_RADIUS = 180
    FPS = 30

    # Colors
    COLOR_BG = (20, 10, 40)
    COLOR_ARENA_BG = (30, 20, 50)
    COLOR_ARENA_BORDER = (80, 60, 120)
    COLOR_ANTIBODY = (0, 180, 255)
    COLOR_ANTIBODY_GLOW = (0, 100, 200)
    COLOR_FIELD = (100, 200, 255)
    BACTERIA_COLORS = {
        1: (255, 80, 80),   # Red
        2: (80, 255, 80),   # Green
        3: (255, 255, 80),  # Yellow
    }
    COLOR_TEXT = (220, 220, 240)

    # Physics & Gameplay
    ANTIBODY_SIZE = 12
    ANTIBODY_LAUNCH_SPEED = 15.0
    ANTIBODY_ROTATION_SPEED = 0.15  # radians per step
    ANTIBODY_DRAG = 0.985
    MIN_VEL_TO_STOP = 0.2
    TIME_DILATION_FACTOR = 0.2
    MAX_EPISODE_STEPS = 5000

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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_level = pygame.font.SysFont("monospace", 24, bold=True)

        self.render_mode = render_mode
        self.level = 1
        self.win_condition_met = False

        # Initialize state variables to avoid attribute errors
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_state = 'aiming'
        self.antibody = {}
        self.bacteria = []
        self.particles = []
        self.prev_space_held = False
        self.stars = []

        # reset() is called here to set up initial state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.win_condition_met:
            self.level += 1

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        self.game_state = 'aiming'
        self.prev_space_held = False

        self.antibody = {
            'pos': self.CENTER.copy(),
            'vel': pygame.math.Vector2(0, 0),
            'angle': -math.pi / 2,  # Pointing up
            'radius': self.ANTIBODY_SIZE
        }

        self._spawn_bacteria()
        self.particles = []
        self.stars = [(self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 3)) for _ in range(100)]

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held_int, shift_held_int = action
        space_held = space_held_int == 1
        # shift_held is unused

        reward = 0
        self.steps += 1

        # --- Handle Input & State Transitions ---
        if self.game_state == 'aiming':
            if movement == 3:  # Left
                self.antibody['angle'] -= self.ANTIBODY_ROTATION_SPEED
            elif movement == 4:  # Right
                self.antibody['angle'] += self.ANTIBODY_ROTATION_SPEED
            self.antibody['angle'] %= (2 * math.pi)

            if space_held and not self.prev_space_held:
                self.game_state = 'launched'
                launch_vec = pygame.math.Vector2()
                launch_vec.from_polar((self.ANTIBODY_LAUNCH_SPEED, math.degrees(-self.antibody['angle'])))
                self.antibody['vel'] = launch_vec

        self.prev_space_held = space_held

        # --- Update Game Logic ---
        time_scale = self.TIME_DILATION_FACTOR if self.game_state == 'aiming' else 1.0

        if self.game_state == 'launched':
            self.antibody['pos'] += self.antibody['vel']
            self.antibody['vel'] *= self.ANTIBODY_DRAG
            if self.antibody['vel'].length() < self.MIN_VEL_TO_STOP:
                self.antibody['vel'] = pygame.math.Vector2(0, 0)
                self.game_state = 'aiming'

        for b in self.bacteria:
            b['timer'] += time_scale
            if b['pattern'] == 'circular':
                angle = b['timer'] * b['speed'] + b['offset']
                b['pos'].x = b['center'].x + math.cos(angle) * b['path_radius']
                b['pos'].y = b['center'].y + math.sin(angle) * b['path_radius']
            elif b['pattern'] == 'linear':
                progress = (math.sin(b['timer'] * b['speed'] + b['offset']) + 1) / 2
                b['pos'] = b['p1'].lerp(b['p2'], progress)

        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

        # --- Collision Detection & Rewards ---
        event_reward = self._handle_collisions_and_rewards()
        reward += event_reward

        # --- Continuous Aiming Reward ---
        if self.game_state == 'aiming' and self.bacteria and not self.game_over:
            nearest_b = min(self.bacteria, key=lambda b: self.antibody['pos'].distance_to(b['pos']))
            angle_to_b = math.atan2(
                -(nearest_b['pos'].y - self.antibody['pos'].y),
                nearest_b['pos'].x - self.antibody['pos'].x
            )
            angle_diff = abs(((self.antibody['angle'] - angle_to_b + math.pi) % (2 * math.pi)) - math.pi)
            aiming_bonus = 0.01 * (1 - angle_diff / math.pi)
            reward += aiming_bonus

        # --- Check Termination Conditions ---
        terminated = self.game_over or not self.bacteria
        truncated = self.steps >= self.MAX_EPISODE_STEPS

        if not self.bacteria and not self.game_over:
            self.win_condition_met = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_collisions_and_rewards(self):
        reward = 0
        # 1. Antibody vs. Arena Boundary (bounce)
        dist_from_center = self.antibody['pos'].distance_to(self.CENTER)
        if dist_from_center > self.ARENA_RADIUS - self.antibody['radius']:
            normal = (self.CENTER - self.antibody['pos']).normalize()
            self.antibody['vel'] = self.antibody['vel'].reflect(normal)
            self.antibody['pos'] = self.CENTER - normal * (self.ARENA_RADIUS - self.antibody['radius'])

        # 2. Bacteria vs. Antibody (Game Over condition)
        if self.game_state == 'aiming':
            for b in self.bacteria:
                if self.antibody['pos'].distance_to(b['pos']) < self.antibody['radius'] + b['radius']:
                    self.game_over = True
                    reward -= 100
                    break
        if self.game_over:
            return reward

        # 3. Launched Antibody vs. Bacteria (Hit condition)
        if self.game_state == 'launched':
            for b in self.bacteria[:]:
                if self.antibody['pos'].distance_to(b['pos']) < self.antibody['radius'] + b['radius']:
                    reward += 1
                    self.score += 1
                    b['hp'] -= 1
                    self._create_particles(b['pos'], self.BACTERIA_COLORS[b['color_id']], 10)

                    normal = (self.antibody['pos'] - b['pos']).normalize()
                    if normal.length() > 0:
                        self.antibody['vel'] = self.antibody['vel'].reflect(normal) * 0.8

                    if b['hp'] <= 0:
                        self.bacteria.remove(b)
                        reward += 10
                        self.score += 10
                        self._create_particles(b['pos'], self.BACTERIA_COLORS[b['color_id']], 30, 3)
                        if not self.bacteria:
                            reward += 50
                            self.score += 50
                    break
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "bacteria_left": len(self.bacteria),
            "game_state": self.game_state,
        }

    def _spawn_bacteria(self):
        self.bacteria = []
        num_bacteria = 2 + self.level
        
        for i in range(num_bacteria):
            level_speed_mod = 1 + (self.level - 1) * 0.05
            hp = 1 + (self.level - 1) // 3
            color_id = min(max(1, hp), 3)

            pattern = self.np_random.choice(['circular', 'linear'])
            
            if pattern == 'circular':
                center = self.CENTER + pygame.math.Vector2(
                    self.np_random.uniform(-50, 50), self.np_random.uniform(-50, 50)
                )
                path_radius = self.np_random.uniform(50, self.ARENA_RADIUS - 50)
                speed = self.np_random.uniform(0.01, 0.02) * level_speed_mod
                offset = self.np_random.uniform(0, 2 * math.pi)
                b = {
                    'pos': pygame.math.Vector2(0, 0),
                    'radius': self.np_random.uniform(8, 12), 'hp': hp, 'max_hp': hp,
                    'color_id': color_id, 'pattern': 'circular', 'center': center,
                    'path_radius': path_radius, 'speed': speed, 'offset': offset, 'timer': 0
                }
            else: # linear
                p1 = self.CENTER + pygame.math.Vector2(
                    self.np_random.uniform(-self.ARENA_RADIUS + 20, self.ARENA_RADIUS - 20),
                    self.np_random.uniform(-self.ARENA_RADIUS + 20, self.ARENA_RADIUS - 20)
                )
                p2 = self.CENTER + pygame.math.Vector2(
                    self.np_random.uniform(-self.ARENA_RADIUS + 20, self.ARENA_RADIUS - 20),
                    self.np_random.uniform(-self.ARENA_RADIUS + 20, self.ARENA_RADIUS - 20)
                )
                speed = self.np_random.uniform(0.02, 0.04) * level_speed_mod
                offset = self.np_random.uniform(0, 2 * math.pi)
                b = {
                    'pos': pygame.math.Vector2(0, 0),
                    'radius': self.np_random.uniform(8, 12), 'hp': hp, 'max_hp': hp,
                    'color_id': color_id, 'pattern': 'linear', 'p1': p1, 'p2': p2,
                    'speed': speed, 'offset': offset, 'timer': 0
                }
            self.bacteria.append(b)

    def _create_particles(self, pos, color, count, speed_mult=1):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = pygame.math.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pos.copy(), 'vel': vel,
                'lifespan': self.np_random.integers(10, 25),
                'color': color, 'radius': self.np_random.uniform(1, 3)
            })

    def _render_game(self):
        for x, y, size in self.stars:
            pygame.draw.rect(self.screen, (100, 100, 120), (x, y, size, size))

        pygame.gfxdraw.filled_circle(self.screen, int(self.CENTER.x), int(self.CENTER.y), self.ARENA_RADIUS, self.COLOR_ARENA_BG)
        pygame.gfxdraw.aacircle(self.screen, int(self.CENTER.x), int(self.CENTER.y), self.ARENA_RADIUS, self.COLOR_ARENA_BORDER)

        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), int(p['radius']))

        for b in self.bacteria:
            color = self.BACTERIA_COLORS[b['color_id']]
            pos = (int(b['pos'].x), int(b['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(b['radius']), color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(b['radius']), tuple(int(c*0.7) for c in color))

        self._draw_antibody()

    def _draw_antibody(self):
        pos, angle, size = self.antibody['pos'], self.antibody['angle'], self.antibody['radius']

        if self.game_state == 'aiming':
            for i in range(int(size / 2), 0, -1):
                alpha = 80 * (1 - i / (size / 2))
                s = pygame.Surface((size*3, size*3), pygame.SRCALPHA)
                pygame.draw.circle(s, (*self.COLOR_ANTIBODY_GLOW, alpha), (size*1.5, size*1.5), size * 1.2 + i)
                self.screen.blit(s, (pos.x - size*1.5, pos.y - size*1.5))
            
            tip = pos + pygame.math.Vector2(size, 0).rotate_rad(angle)
            for i in range(3):
                offset_angle = (i - 1) * 0.4
                start_angle, end_angle = -angle - 0.5 + offset_angle, -angle + 0.5 + offset_angle
                rect = pygame.Rect(tip.x - 30, tip.y - 30, 60, 60)
                pygame.draw.arc(self.screen, self.COLOR_FIELD, rect, start_angle, end_angle, 1)

        p1 = pos + pygame.math.Vector2(size, 0).rotate_rad(angle)
        p2 = pos + pygame.math.Vector2(-size/2, size*0.8).rotate_rad(angle)
        p3 = pos + pygame.math.Vector2(-size/2, -size*0.8).rotate_rad(angle)
        points = [(int(p.x), int(p.y)) for p in [p1, p2, p3]]
        
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ANTIBODY)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ANTIBODY)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        bact_text = self.font_ui.render(f"BACTERIA: {len(self.bacteria)}", True, self.COLOR_TEXT)
        self.screen.blit(bact_text, (self.WIDTH - bact_text.get_width() - 10, 10))

        level_text = self.font_level.render(f"LEVEL {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.WIDTH/2 - level_text.get_width()/2, 10))
    
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and is not part of the Gymnasium environment API.
    # It will be ignored by the automated tests.
    render_human = True
    if render_human:
        # Re-enable video driver for human play
        os.environ["SDL_VIDEODRIVER"] = "x11" 
        import pygame
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    if render_human:
        pygame.display.set_caption("Antibody Annihilation")
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        movement_action = 0 # No-op for movement
        space_action = 0
        shift_action = 0

        if render_human:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                movement_action = 3
            if keys[pygame.K_RIGHT]:
                movement_action = 4
            if keys[pygame.K_SPACE]:
                space_action = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift_action = 1
        else: # Simple agent for testing without human input
             action = env.action_space.sample()
             movement_action, space_action, shift_action = action.tolist()

        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            obs, info = env.reset()
            total_reward = 0
            if not render_human: # break loop if not interactive
                done = True

        if render_human:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(GameEnv.FPS)

    env.close()