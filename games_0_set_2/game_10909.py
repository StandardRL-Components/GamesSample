import gymnasium as gym
import os
import pygame
import numpy as np
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame.init()
pygame.font.init()

from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw


class GameEnv(gym.Env):
    """
    Gymnasium environment for 'Quark Fusion'.

    **Objective:** Control three quarks to activate all five gluons within a time limit.
    **Visuals:** A clean, geometric style with vibrant particle effects on a dark background.
    **Mechanics:**
    - Quarks are controlled simultaneously.
    - Touching a gluon with a quark activates it.
    - Activated gluons can trigger a chain reaction with nearby gluons.
    **Action Space:** MultiDiscrete([5, 2, 2])
    - action[0]: Movement direction (0:None, 1:Up, 2:Down, 3:Left, 4:Right)
    - action[1]: Apply movement to Quark 2 (Green)
    - action[2]: Apply movement to Quark 3 (Blue)
    (Movement is always applied to Quark 1)
    **Reward Structure:**
    - +0.1 for each gluon activated.
    - +1.0 bonus for activating multiple gluons in a single step.
    - +100 for winning (all gluons activated).
    - -100 for losing (time runs out).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control three quarks simultaneously to activate all five gluons by touching them before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys to move. The red quark always moves. Hold space to also move the green quark, and hold shift to also move the blue quark."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 50
    DT = 1.0 / FPS
    MAX_TIME = 60.0  # seconds
    MAX_STEPS = int(MAX_TIME * FPS)

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_QUARK_R = (255, 50, 50)
    COLOR_QUARK_G = (50, 255, 50)
    COLOR_QUARK_B = (50, 100, 255)
    COLOR_GLUON_INACTIVE = (255, 200, 0)
    COLOR_GLUON_ACTIVE = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TIMER_WARN = (255, 100, 100)

    # Game element properties
    QUARK_SIZE = 12
    GLUON_RADIUS = 15
    GLUON_COUNT = 5
    CHAIN_REACTION_RADIUS = 75
    ACTIVATION_RING_SPEED = 150

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # --- Game State Variables ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = self.MAX_TIME
        
        self.quarks = []
        self.gluons = []
        self.particles = []
        self.rings = []

        self.activated_in_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = self.MAX_TIME
        
        self.particles.clear()
        self.rings.clear()
        self._initialize_quarks()
        self._initialize_gluons()

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        if self.game_over:
            obs = self._get_observation()
            info = self._get_info()
            return obs, 0.0, True, False, info

        self.steps += 1
        self.timer -= self.DT
        self.activated_in_step = 0

        self._handle_input(action)
        self._update_quarks()
        self._update_effects()
        self._handle_collisions_and_activations()
        
        reward = self._calculate_reward()
        self.score += reward

        terminated = self._check_termination()
        truncated = False # This environment does not truncate based on steps
        
        # Add terminal rewards
        if terminated and not self.game_over:
            if self.timer <= 0:
                reward -= 100.0 # Loss
                self.score -= 100.0
            else: # Win
                reward += 100.0
                self.score += 100.0
            self.game_over = True

        obs = self._get_observation()
        info = self._get_info()
        
        return (
            obs,
            reward,
            terminated,
            truncated,
            info
        )

    def _initialize_quarks(self):
        self.quarks = [
            {'pos': np.array([50.0, 50.0]), 'vel': np.zeros(2), 'color': self.COLOR_QUARK_R, 'speed': 100.0},
            {'pos': np.array([self.WIDTH - 50.0, 50.0]), 'vel': np.zeros(2), 'color': self.COLOR_QUARK_G, 'speed': 150.0},
            {'pos': np.array([50.0, self.HEIGHT - 50.0]), 'vel': np.zeros(2), 'color': self.COLOR_QUARK_B, 'speed': 200.0}
        ]

    def _initialize_gluons(self):
        self.gluons.clear()
        # Place gluons in a way that allows for chain reactions
        center_x, center_y = self.WIDTH / 2, self.HEIGHT / 2
        self.gluons.append({'pos': np.array([center_x, center_y]), 'activated': False, 'radius': self.GLUON_RADIUS})
        self.gluons.append({'pos': np.array([center_x - 60, center_y - 60]), 'activated': False, 'radius': self.GLUON_RADIUS})
        self.gluons.append({'pos': np.array([center_x + 60, center_y - 60]), 'activated': False, 'radius': self.GLUON_RADIUS})
        self.gluons.append({'pos': np.array([center_x - 60, center_y + 60]), 'activated': False, 'radius': self.GLUON_RADIUS})
        self.gluons.append({'pos': np.array([center_x + 60, center_y + 60]), 'activated': False, 'radius': self.GLUON_RADIUS})

    def _handle_input(self, action):
        movement_idx, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        direction = np.zeros(2)
        if movement_idx == 1: direction[1] = -1  # Up
        elif movement_idx == 2: direction[1] = 1   # Down
        elif movement_idx == 3: direction[0] = -1  # Left
        elif movement_idx == 4: direction[0] = 1   # Right

        # Quark 1 (Red) always responds to the direction
        self.quarks[0]['vel'] = direction * self.quarks[0]['speed']
        
        # Quark 2 (Green) responds if space is held
        self.quarks[1]['vel'] = direction * self.quarks[1]['speed'] if space_held else np.zeros(2)
        
        # Quark 3 (Blue) responds if shift is held
        self.quarks[2]['vel'] = direction * self.quarks[2]['speed'] if shift_held else np.zeros(2)

    def _update_quarks(self):
        for q in self.quarks:
            q['pos'] += q['vel'] * self.DT
            # Boundary checks
            q['pos'][0] = np.clip(q['pos'][0], self.QUARK_SIZE / 2, self.WIDTH - self.QUARK_SIZE / 2)
            q['pos'][1] = np.clip(q['pos'][1], self.QUARK_SIZE / 2, self.HEIGHT - self.QUARK_SIZE / 2)
            
            # Create trail particles
            if np.linalg.norm(q['vel']) > 0:
                if self.steps % 3 == 0:
                    trail_pos = q['pos'].copy()
                    trail_vel = (self.np_random.random(2) - 0.5) * 10
                    self._create_particles(1, trail_pos, q['color'], 0.5, trail_vel, 2)

    def _update_effects(self):
        # Update particles
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'] += p['vel'] * self.DT
            p['lifetime'] -= self.DT
            p['radius'] = max(0, p['initial_radius'] * (p['lifetime'] / p['initial_lifetime']))
        
        # Update rings
        self.rings = [r for r in self.rings if r['progress'] < 1.0]
        for r in self.rings:
            r['progress'] += self.DT * 0.8 # Rings last 1.25s

    def _handle_collisions_and_activations(self):
        newly_activated_this_frame = []

        # Quark-Gluon collisions
        for gluon in self.gluons:
            if not gluon['activated']:
                for quark in self.quarks:
                    dist = np.linalg.norm(quark['pos'] - gluon['pos'])
                    if dist < gluon['radius'] + self.QUARK_SIZE / 2:
                        if not gluon['activated']:
                            newly_activated_this_frame.append(gluon)
                            gluon['activated'] = True
                        break # A gluon can only be activated once per frame

        # Chain reactions
        chain_candidates = list(newly_activated_this_frame)
        while chain_candidates:
            activator = chain_candidates.pop(0)
            for other_gluon in self.gluons:
                if not other_gluon['activated']:
                    dist = np.linalg.norm(activator['pos'] - other_gluon['pos'])
                    if dist < self.CHAIN_REACTION_RADIUS:
                        other_gluon['activated'] = True
                        newly_activated_this_frame.append(other_gluon)
                        chain_candidates.append(other_gluon)
        
        # Create visual effects for activations
        for gluon in newly_activated_this_frame:
            self.activated_in_step += 1
            self._create_particles(30, gluon['pos'], self.COLOR_GLUON_ACTIVE, 1.0)
            self.rings.append({'pos': gluon['pos'], 'progress': 0.0, 'max_radius': self.CHAIN_REACTION_RADIUS})

    def _calculate_reward(self):
        reward = 0.0
        if self.activated_in_step > 0:
            reward += self.activated_in_step * 0.1
        if self.activated_in_step > 1:
            reward += 1.0 # Bonus for simultaneous/chain activation
        return reward

    def _check_termination(self):
        if self.timer <= 0:
            return True
        if all(g['activated'] for g in self.gluons):
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        activated_count = sum(1 for g in self.gluons if g['activated'])
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "activated_gluons": activated_count
        }

    def _render_game(self):
        # Render rings
        for r in self.rings:
            current_radius = int(r['progress'] * r['max_radius'])
            alpha = int(max(0, 255 * (1 - r['progress']**0.5)))
            if current_radius > 0:
                pygame.gfxdraw.aacircle(self.screen, int(r['pos'][0]), int(r['pos'][1]), current_radius, self.COLOR_GLUON_ACTIVE + (alpha,))

        # Render particles
        for p in self.particles:
            alpha = int(max(0, 255 * (p['lifetime'] / p['initial_lifetime'])))
            color = p['color'] + (alpha,)
            if p['radius'] > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)

        # Render gluons
        for g in self.gluons:
            pos_int = (int(g['pos'][0]), int(g['pos'][1]))
            color = self.COLOR_GLUON_ACTIVE if g['activated'] else self.COLOR_GLUON_INACTIVE
            glow_color = color + (64,)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], g['radius'] + 4, glow_color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], g['radius'] + 4, glow_color)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], g['radius'], color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], g['radius'], color)

        # Render quarks
        for q in self.quarks:
            pos_int = (int(q['pos'][0]), int(q['pos'][1]))
            size = self.QUARK_SIZE
            rect = pygame.Rect(pos_int[0] - size / 2, pos_int[1] - size / 2, size, size)
            glow_rect = rect.inflate(size, size)
            
            glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, q['color'] + (32,), glow_surf.get_rect(), border_radius=int(size*0.8))
            pygame.draw.rect(glow_surf, q['color'] + (48,), glow_surf.get_rect().inflate(-size*0.5, -size*0.5), border_radius=int(size*0.6))
            self.screen.blit(glow_surf, glow_rect.topleft)
            
            pygame.draw.rect(self.screen, q['color'], rect, border_radius=3)
            
    def _render_ui(self):
        # Timer
        timer_color = self.COLOR_TEXT if self.timer > 10 else self.COLOR_TIMER_WARN
        timer_text = self.font_large.render(f"{self.timer:.1f}", True, timer_color)
        self.screen.blit(timer_text, (20, 10))

        # Activated Gluons
        activated_count = sum(1 for g in self.gluons if g['activated'])
        gluon_text = self.font_large.render(f"{activated_count} / {self.GLUON_COUNT}", True, self.COLOR_TEXT)
        self.screen.blit(gluon_text, (self.WIDTH - gluon_text.get_width() - 20, 10))
        
        # Game Over Message
        if self.game_over:
            won = all(g['activated'] for g in self.gluons)
            msg = "FUSION COMPLETE" if won else "TIME EXPIRED"
            color = self.COLOR_QUARK_G if won else self.COLOR_QUARK_R
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, count, pos, color, lifetime, base_vel=None, radius=4):
        for _ in range(count):
            if base_vel is None:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(20, 80)
                vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            else:
                vel = base_vel + (self.np_random.random(2) - 0.5) * 20
            
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifetime': self.np_random.uniform(lifetime * 0.5, lifetime),
                'initial_lifetime': lifetime,
                'color': color,
                'radius': radius,
                'initial_radius': radius
            })

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == "__main__":
    # --- Human Playable Demo ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human display
    pygame.display.set_caption("Quark Fusion")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0.0
    
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0.0

        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}. Info: {info}")
            print("Press 'R' to restart.")

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to convert it back to a Pygame surface to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(GameEnv.FPS)

    env.close()