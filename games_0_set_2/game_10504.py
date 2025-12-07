import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:29:39.870561
# Source Brief: brief_00504.md
# Brief Index: 504
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# --- Constants ---
WIDTH, HEIGHT = 640, 400
FPS = 60
WIN_TIME_SECONDS = 60
MAX_STEPS = WIN_TIME_SECONDS * FPS
NUM_BALLS = 5

# --- Colors (Vibrant & High Contrast) ---
COLOR_BG = (15, 18, 23)
COLOR_GROUND = (40, 45, 55)
COLOR_BALL_NORMAL = (255, 70, 70)
COLOR_BALL_TRANSFORMED = (255, 220, 50)
COLOR_ORB = (70, 255, 120)
COLOR_FORCE_FIELD = (100, 180, 255)
COLOR_TEXT = (220, 220, 240)
COLOR_SELECTED = (255, 255, 255)

# --- Physics & Gameplay ---
GRAVITY_NORMAL = 0.15
GRAVITY_TRANSFORMED = 0.075
BALL_SPEED_ADJUST = 0.4
WALL_DAMPING = 0.85
FORCE_FIELD_STRENGTH = 0.3
FORCE_FIELD_RADIUS = 60.0
FORCE_FIELD_LIFESPAN = 0.5 * FPS  # 30 frames
ORB_SPAWN_INTERVAL = 2 * FPS  # 120 frames
TRANSFORMATION_ORB_COUNT = 20


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player juggles five bouncing balls.
    The goal is to keep all balls on-screen for 60 seconds by adjusting their
    vertical speeds. Balls create repulsive force fields on collision and can be
    transformed by collecting orbs, altering their physics.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": FPS}

    game_description = (
        "Juggle five bouncing balls by applying vertical force. Collect orbs to transform balls "
        "and keep them on screen for 60 seconds to win."
    )
    user_guide = (
        "Controls: ←→ to select a ball, ↑↓ to apply vertical force. "
        "Keep all balls from falling off the screen."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(HEIGHT, WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont('Consolas', 24, bold=True)
        self.font_small = pygame.font.SysFont('Consolas', 14, bold=True)

        # Initialize state variables to ensure they exist
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.balls = []
        self.orbs = []
        self.force_fields = []
        self.selected_ball_index = 0
        self.orb_spawn_timer = 0
        
        # This call ensures a valid state for the validator
        # A seed is needed for np_random to be initialized
        self.reset(seed=0)


    def _initialize_game_state(self):
        """Initializes or resets all game state variables."""
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.selected_ball_index = 0
        self.orb_spawn_timer = self.np_random.integers(0, ORB_SPAWN_INTERVAL)
        self.orbs = []
        self.force_fields = []

        self.balls = []
        for i in range(NUM_BALLS):
            self.balls.append({
                'pos': pygame.Vector2(WIDTH * (i + 1) / (NUM_BALLS + 1), HEIGHT / 3),
                'vel': pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-2, 0)),
                'radius': 12,
                'orb_count': 0,
                'is_transformed': False,
                'color': COLOR_BALL_NORMAL,
                'mass': 1.0,
            })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_game_state()
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_input(action)
        
        self.steps += 1
        
        self._update_orbs()
        self._update_force_fields()
        self._update_balls()
        
        orb_collection_reward = self._handle_collisions()
        self.score += orb_collection_reward
        
        reward = 0.1 + orb_collection_reward  # Base survival reward + event reward
        
        terminated = False
        if any(ball['pos'].y > HEIGHT + ball['radius'] for ball in self.balls):
            # Sound: Game over failure
            terminated = True
            reward = -100.0
            self.game_over = True
        elif self.steps >= MAX_STEPS:
            # Sound: Game win success
            terminated = True
            reward = 100.0
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        """Processes the agent's action."""
        movement = action[0]
        
        if movement == 3:  # Left
            # Sound: UI select
            self.selected_ball_index = (self.selected_ball_index - 1 + NUM_BALLS) % NUM_BALLS
        elif movement == 4:  # Right
            # Sound: UI select
            self.selected_ball_index = (self.selected_ball_index + 1) % NUM_BALLS
        
        selected_ball = self.balls[self.selected_ball_index]
        speed_adjust = BALL_SPEED_ADJUST / 2 if selected_ball['is_transformed'] else BALL_SPEED_ADJUST
        
        if movement == 1:  # Up
            selected_ball['vel'].y -= speed_adjust
        elif movement == 2:  # Down
            selected_ball['vel'].y += speed_adjust

    def _update_orbs(self):
        """Handles spawning and pruning of orbs."""
        self.orb_spawn_timer += 1
        if self.orb_spawn_timer > ORB_SPAWN_INTERVAL:
            # Sound: Orb spawn
            self.orb_spawn_timer = 0
            self.orbs.append(pygame.Vector2(self.np_random.uniform(50, WIDTH - 50), self.np_random.uniform(100, HEIGHT - 100)))
            if len(self.orbs) > 10:
                self.orbs.pop(0)

    def _update_force_fields(self):
        """Updates lifespan of force fields and removes expired ones."""
        self.force_fields = [ff for ff in self.force_fields if ff['lifespan'] > 0]
        for ff in self.force_fields:
            ff['lifespan'] -= 1

    def _update_balls(self):
        """Applies physics to all balls."""
        for ball in self.balls:
            gravity = GRAVITY_TRANSFORMED if ball['is_transformed'] else GRAVITY_NORMAL
            ball['vel'].y += gravity
            
            for ff in self.force_fields:
                dist_vec = ball['pos'] - ff['pos']
                dist = dist_vec.length()
                if 0 < dist < FORCE_FIELD_RADIUS:
                    repulsion_strength = FORCE_FIELD_STRENGTH * (1 - dist / FORCE_FIELD_RADIUS)
                    force = dist_vec.normalize() * repulsion_strength
                    ball['vel'] += force

            ball['pos'] += ball['vel']

            if ball['pos'].x < ball['radius'] or ball['pos'].x > WIDTH - ball['radius']:
                ball['vel'].x *= -WALL_DAMPING
                ball['pos'].x = np.clip(ball['pos'].x, ball['radius'], WIDTH - ball['radius'])
            if ball['pos'].y < ball['radius']:
                # Sound: Ball bounce soft
                ball['vel'].y *= -WALL_DAMPING
                ball['pos'].y = ball['radius']
            if ball['pos'].y > HEIGHT - 5 - ball['radius']:
                # Sound: Ball bounce hard
                ball['vel'].y *= -WALL_DAMPING
                ball['pos'].y = HEIGHT - 5 - ball['radius']

    def _handle_collisions(self):
        """Handles ball-orb and ball-ball collisions."""
        orb_reward = 0.0
        
        # Ball-Orb collisions
        collected_orbs_indices = []
        for ball in self.balls:
            for i, orb_pos in enumerate(self.orbs):
                if i not in collected_orbs_indices and (ball['pos'] - orb_pos).length() < ball['radius'] + 5:
                    # Sound: Orb collect
                    collected_orbs_indices.append(i)
                    ball['orb_count'] += 1
                    orb_reward += 1.0
                    if not ball['is_transformed'] and ball['orb_count'] >= TRANSFORMATION_ORB_COUNT:
                        # Sound: Transformation
                        ball['is_transformed'] = True
                        ball['radius'] *= 2
                        ball['mass'] *= 4
                        ball['color'] = COLOR_BALL_TRANSFORMED
        self.orbs = [orb for i, orb in enumerate(self.orbs) if i not in collected_orbs_indices]
        
        # Ball-Ball collisions
        for i in range(NUM_BALLS):
            for j in range(i + 1, NUM_BALLS):
                b1, b2 = self.balls[i], self.balls[j]
                dist_vec = b1['pos'] - b2['pos']
                dist = dist_vec.length()
                if dist < b1['radius'] + b2['radius']:
                    # Sound: Ball collision / Force field spawn
                    self.force_fields.append({'pos': (b1['pos'] + b2['pos']) / 2, 'lifespan': FORCE_FIELD_LIFESPAN})
                    
                    if dist > 0:
                        overlap = (b1['radius'] + b2['radius']) - dist
                        push_vec = dist_vec.normalize() * overlap
                        b1['pos'] += push_vec * (b2['mass'] / (b1['mass'] + b2['mass']))
                        b2['pos'] -= push_vec * (b1['mass'] / (b1['mass'] + b2['mass']))
                        
                        normal = dist_vec.normalize()
                        v1n = b1['vel'].dot(normal)
                        v2n = b2['vel'].dot(normal)
                        
                        new_v1n = (v1n * (b1['mass'] - b2['mass']) + 2 * b2['mass'] * v2n) / (b1['mass'] + b2['mass'])
                        new_v2n = (v2n * (b2['mass'] - b1['mass']) + 2 * b1['mass'] * v1n) / (b1['mass'] + b2['mass'])
                        
                        b1['vel'] += (new_v1n - v1n) * normal
                        b2['vel'] += (new_v2n - v2n) * normal
        return orb_reward

    def _get_observation(self):
        self.screen.fill(COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.draw.rect(self.screen, COLOR_GROUND, (0, HEIGHT - 5, WIDTH, 5))

        for ff in self.force_fields:
            alpha = int(200 * (ff['lifespan'] / FORCE_FIELD_LIFESPAN)**2)
            if alpha > 0:
                self._draw_fading_circle(self.screen, COLOR_FORCE_FIELD, ff['pos'], int(FORCE_FIELD_RADIUS), alpha)

        for orb_pos in self.orbs:
            self._draw_glowing_circle(self.screen, COLOR_ORB, orb_pos, 5, 0.6)

        for i, ball in enumerate(self.balls):
            pos = (int(ball['pos'].x), int(ball['pos'].y))
            radius = int(ball['radius'])
            
            self._draw_glowing_circle(self.screen, ball['color'], pos, radius, 0.5)
            
            if i == self.selected_ball_index:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius + 4, COLOR_SELECTED)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius + 5, COLOR_SELECTED)

            if ball['orb_count'] > 0:
                text_surf = self.font_small.render(str(ball['orb_count']), True, COLOR_TEXT)
                text_rect = text_surf.get_rect(center=(pos[0], pos[1] - radius - 12))
                self.screen.blit(text_surf, text_rect)

    def _draw_glowing_circle(self, surface, color, center, radius, glow_factor):
        center_int = (int(center[0]), int(center[1]))
        glow_radius = int(radius * (1 + glow_factor))
        
        # Create a temporary surface for the glow effect
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        
        # Draw concentric circles with decreasing alpha for the glow
        for i in range(glow_radius - radius, 0, -2):
            alpha = int(100 * (1 - (i / (glow_radius - radius)))**2)
            glow_color = (*color, alpha)
            pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), radius + i)
            
        surface.blit(temp_surf, (center_int[0] - glow_radius, center_int[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Draw the main, solid circle on top
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], radius, color)
        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], radius, color)

    def _draw_fading_circle(self, surface, color, center, radius, alpha):
        center_int = (int(center[0]), int(center[1]))
        temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (*color, int(alpha/3)), (radius, radius), radius)
        pygame.draw.circle(temp_surf, (*color, alpha), (radius, radius), radius, width=max(1, int(radius/10)))
        surface.blit(temp_surf, (center_int[0] - radius, center_int[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        time_left = max(0, WIN_TIME_SECONDS - self.steps / FPS)
        timer_text = f"TIME: {time_left:.1f}"
        text_surf = self.font_main.render(timer_text, True, COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))
        
        score_text = f"SCORE: {self.score:.1f}"
        score_surf = self.font_main.render(score_text, True, COLOR_TEXT)
        score_rect = score_surf.get_rect(topright=(WIDTH - 10, 10))
        self.screen.blit(score_surf, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, WIN_TIME_SECONDS - self.steps / FPS),
            "balls_on_screen": len([b for b in self.balls if b['pos'].y <= HEIGHT])
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (HEIGHT, WIDTH, 3), f"Obs shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (HEIGHT, WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (HEIGHT, WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # Set a non-dummy driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    terminated = False
    
    # --- Manual Play Setup ---
    pygame.display.set_caption("Ball Juggler")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    # Action state
    current_action = np.array([0, 0, 0])

    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Q: Quit")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False
            
        keys = pygame.key.get_pressed()
        
        # Reset movement action
        current_action[0] = 0
        if keys[pygame.K_UP]:
            current_action[0] = 1
        elif keys[pygame.K_DOWN]:
            current_action[0] = 2
        elif keys[pygame.K_LEFT]:
            current_action[0] = 3
        elif keys[pygame.K_RIGHT]:
            current_action[0] = 4

        obs, reward, terminated, truncated, info = env.step(current_action)

        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
        
        # Render to screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(FPS)
        
    env.close()