import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:06:52.507044
# Source Brief: brief_00888.md
# Brief Index: 888
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Frogger River Crossing Environment

    The agent controls a frog trying to cross a river with varying horizontal currents.
    The goal is to reach the safety of the top bank.

    Action Space: MultiDiscrete([5, 2, 2])
    - Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - Powerful Kick (0=released, 1=held)
    - Unused (0=released, 1=held)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Rewards:
    - +100 for reaching the top bank.
    - -100 for being swept off-screen.
    - +1 for landing on a lily pad.
    - +0.1 per pixel moved upwards.
    - -0.01 per pixel moved downwards.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Guide a frog across a treacherous river with powerful currents. "
        "Land on lily pads for a respite and reach the far bank to win."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to swim. Press space for a powerful forward kick to fight the current."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30

    # Colors
    COLOR_BG_WATER_DEEP = (20, 40, 90)
    COLOR_BG_WATER_SHALLOW = (40, 80, 150)
    COLOR_BANK = (60, 140, 60)
    COLOR_FINISH_LINE = (90, 180, 90)
    COLOR_FROG = (100, 220, 100)
    COLOR_FROG_GLOW = (180, 255, 180)
    COLOR_LILY_PAD = (20, 100, 20)
    COLOR_LILY_PAD_HIGHLIGHT = (30, 150, 30)
    COLOR_RIPPLE = (200, 220, 255)
    COLOR_FLOW_LINE = (100, 120, 180)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0)

    # Physics
    STROKE_FORCE = 0.3
    KICK_FORCE = 1.8
    KICK_COOLDOWN_STEPS = 45  # 1.5 seconds
    DRAG_COEFFICIENT = 0.96
    MAX_VELOCITY = 5.0
    FROG_RADIUS = 12
    LILY_PAD_RADIUS = 20

    # Game Rules
    MAX_STEPS = 2000
    WIN_Y_COORD = HEIGHT - 30
    NUM_LILY_PADS = 8
    NUM_FLOW_LINES = 100

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 30)

        self.render_mode = render_mode
        self._initialize_state_variables()

    def _initialize_state_variables(self):
        """Initializes all state variables. Called by __init__ and reset."""
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.frog_pos = np.array([self.WIDTH / 2, 20.0], dtype=np.float64)
        self.frog_vel = np.array([0.0, 0.0], dtype=np.float64)
        self.frog_angle = -90.0 # Pointing up

        self.kick_cooldown = 0
        self.on_lily_pad = False
        self.was_on_lily_pad = False

        self.lily_pads = []
        self.ripples = []
        self.flow_lines = []
        self.base_current_speed = 1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state_variables()

        # Generate lily pads
        pad_y_zones = np.linspace(80, self.HEIGHT - 80, self.NUM_LILY_PADS)
        for i in range(self.NUM_LILY_PADS):
            pad_x = self.np_random.uniform(self.LILY_PAD_RADIUS, self.WIDTH - self.LILY_PAD_RADIUS)
            pad_y = self.np_random.uniform(pad_y_zones[i] - 10, pad_y_zones[i] + 10)
            self.lily_pads.append({'pos': np.array([pad_x, pad_y]), 'radius': self.LILY_PAD_RADIUS})

        # Generate flow lines for water visualization
        for _ in range(self.NUM_FLOW_LINES):
            self.flow_lines.append({
                'pos': np.array([self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)]),
                'length': self.np_random.uniform(5, 15),
                'thickness': self.np_random.choice([1, 2])
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        prev_y = self.frog_pos[1]
        self.was_on_lily_pad = self.on_lily_pad

        self._handle_actions(action)
        self._update_physics()
        self._update_effects()
        self._check_collisions_and_boundaries()

        reward = self._calculate_reward(prev_y)
        self.score += reward
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated or truncated:
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_actions(self, action):
        movement, space_held, _ = action
        stroke_force_vec = np.array([0.0, 0.0])
        action_taken = False

        if movement == 1:  # Up
            stroke_force_vec[1] += self.STROKE_FORCE
            action_taken = True
        elif movement == 2:  # Down
            stroke_force_vec[1] -= self.STROKE_FORCE
            action_taken = True
        elif movement == 3:  # Left
            stroke_force_vec[0] -= self.STROKE_FORCE
            action_taken = True
        elif movement == 4:  # Right
            stroke_force_vec[0] += self.STROKE_FORCE
            action_taken = True

        if action_taken:
            self._spawn_ripple(self.frog_pos, 5, 20, 0.5)
            # Sfx: water splash
            
        if space_held and self.kick_cooldown == 0:
            # Kick is always "forward" (up the river)
            stroke_force_vec[1] += self.KICK_FORCE
            self.kick_cooldown = self.KICK_COOLDOWN_STEPS
            self._spawn_ripple(self.frog_pos, 10, 30, 1.0)
            # Sfx: big water kick

        self.frog_vel += stroke_force_vec

    def _update_physics(self):
        # Update cooldowns
        if self.kick_cooldown > 0:
            self.kick_cooldown -= 1

        # Check if on a lily pad
        self.on_lily_pad = False
        for pad in self.lily_pads:
            if np.linalg.norm(self.frog_pos - pad['pos']) < pad['radius']:
                self.on_lily_pad = True
                self.frog_vel *= 0.8 # Stick to the pad
                break

        # Apply current and drag if in water
        if not self.on_lily_pad:
            current_vel = self._get_current_at_pos(self.frog_pos)
            self.frog_vel[0] += current_vel
            self.frog_vel *= self.DRAG_COEFFICIENT

        # Clamp velocity
        velocity_norm = np.linalg.norm(self.frog_vel)
        if velocity_norm > self.MAX_VELOCITY:
            self.frog_vel = self.frog_vel * (self.MAX_VELOCITY / velocity_norm)

        # Update position
        self.frog_pos += self.frog_vel

        # Update frog angle for visual flair
        if np.linalg.norm(self.frog_vel) > 0.1:
            self.frog_angle = math.degrees(math.atan2(-self.frog_vel[1], self.frog_vel[0])) - 90

    def _get_current_at_pos(self, pos):
        # Current speed increases over time
        self.base_current_speed = 1.0 + (self.steps // 200) * 0.05
        # Current is stronger in the middle of the river, weaker at the banks
        y_factor = (pos[1] / self.HEIGHT) * (1 - pos[1] / self.HEIGHT) * 4  # Parabolic profile
        return self.base_current_speed * y_factor

    def _check_collisions_and_boundaries(self):
        # Prevent frog from going through lily pads
        for pad in self.lily_pads:
            dist_vec = self.frog_pos - pad['pos']
            dist = np.linalg.norm(dist_vec)
            if dist < pad['radius'] + self.FROG_RADIUS / 2:
                overlap = pad['radius'] + self.FROG_RADIUS / 2 - dist
                self.frog_pos += (dist_vec / dist) * overlap
                self.frog_vel -= (dist_vec / dist) * np.dot(self.frog_vel, dist_vec / dist)

    def _calculate_reward(self, prev_y):
        reward = 0.0
        
        # Reward for vertical progress
        delta_y = self.frog_pos[1] - prev_y
        if delta_y > 0:
            reward += 0.1 * delta_y
        else:
            reward += 0.01 * delta_y # Penalty for moving down

        # Reward for landing on a lily pad
        if self.on_lily_pad and not self.was_on_lily_pad:
            reward += 1.0
            # Sfx: safe landing

        # Win condition
        if self.frog_pos[1] >= self.WIN_Y_COORD:
            reward += 100

        # Loss conditions
        if not (0 < self.frog_pos[0] < self.WIDTH and 0 < self.frog_pos[1] < self.HEIGHT):
            reward -= 100

        return reward

    def _check_termination(self):
        # Win condition
        if self.frog_pos[1] >= self.WIN_Y_COORD:
            return True

        # Loss conditions
        if not (0 < self.frog_pos[0] < self.WIDTH and 0 < self.frog_pos[1] < self.HEIGHT):
            return True
        
        return False

    def _spawn_ripple(self, pos, start_radius, max_radius, life):
        self.ripples.append({
            'pos': pos.copy(),
            'radius': start_radius,
            'max_radius': max_radius,
            'life': life,
            'max_life': life
        })

    def _update_effects(self):
        # Update ripples
        new_ripples = []
        for r in self.ripples:
            r['life'] -= 1 / self.FPS
            r['radius'] += (r['max_radius'] / r['max_life']) / self.FPS
            if r['life'] > 0:
                new_ripples.append(r)
        self.ripples = new_ripples

        # Update flow lines
        for line in self.flow_lines:
            current_vel = self._get_current_at_pos(line['pos'])
            line['pos'][0] += current_vel
            if line['pos'][0] > self.WIDTH:
                line['pos'][0] = 0
                line['pos'][1] = self.np_random.uniform(0, self.HEIGHT)

    def _get_observation(self):
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "frog_pos": self.frog_pos,
            "on_lily_pad": self.on_lily_pad,
        }

    def _render_game(self):
        # Draw water gradient
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_WATER_DEEP[0] * (1 - interp) + self.COLOR_BG_WATER_SHALLOW[0] * interp,
                self.COLOR_BG_WATER_DEEP[1] * (1 - interp) + self.COLOR_BG_WATER_SHALLOW[1] * interp,
                self.COLOR_BG_WATER_DEEP[2] * (1 - interp) + self.COLOR_BG_WATER_SHALLOW[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        # Draw river banks
        pygame.draw.rect(self.screen, self.COLOR_BANK, (0, 0, self.WIDTH, 30))
        pygame.draw.rect(self.screen, self.COLOR_BANK, (0, self.HEIGHT - 30, self.WIDTH, 30))
        pygame.draw.rect(self.screen, self.COLOR_FINISH_LINE, (0, self.HEIGHT - 35, self.WIDTH, 5))

        # Draw flow lines
        for line in self.flow_lines:
            start_pos = (int(line['pos'][0]), int(line['pos'][1]))
            end_pos = (int(line['pos'][0] + line['length']), int(line['pos'][1]))
            pygame.draw.line(self.screen, self.COLOR_FLOW_LINE, start_pos, end_pos, line['thickness'])

        # Draw ripples
        for r in self.ripples:
            alpha = int(255 * (r['life'] / r['max_life']))
            if alpha > 0:
                color = (*self.COLOR_RIPPLE, alpha)
                pygame.gfxdraw.aacircle(self.screen, int(r['pos'][0]), int(r['pos'][1]), int(r['radius']), color)

        # Draw lily pads
        for pad in self.lily_pads:
            pos = (int(pad['pos'][0]), int(pad['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], pad['radius'], self.COLOR_LILY_PAD)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], pad['radius'], self.COLOR_LILY_PAD_HIGHLIGHT)
            # Add a little detail
            cut_angle = math.radians(45)
            p1 = (pos[0] + pad['radius'] * math.cos(cut_angle), pos[1] + pad['radius'] * math.sin(cut_angle))
            pygame.draw.line(self.screen, self.COLOR_LILY_PAD_HIGHLIGHT, pos, p1)


        # Draw frog
        self._render_frog()

    def _render_frog(self):
        pos = (int(self.frog_pos[0]), int(self.frog_pos[1]))

        # Glow effect
        glow_radius = int(self.FROG_RADIUS * 1.5)
        if self.kick_cooldown > 0:
            # Pulsing glow after kick
            pulse = abs(math.sin(self.steps * 0.5))
            glow_radius = int(self.FROG_RADIUS * (1.5 + pulse * 0.5))
        
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        glow_color = (*self.COLOR_FROG_GLOW, 50)
        pygame.draw.circle(s, glow_color, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Body
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.FROG_RADIUS, self.COLOR_FROG)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.FROG_RADIUS, (0,0,0))

        # Eyes
        angle_rad = math.radians(self.frog_angle)
        for i in [-1, 1]:
            eye_base_angle = math.radians(45 * i)
            eye_x = pos[0] + (self.FROG_RADIUS * 0.7) * math.cos(angle_rad + eye_base_angle)
            eye_y = pos[1] + (self.FROG_RADIUS * 0.7) * math.sin(angle_rad + eye_base_angle)
            pygame.draw.circle(self.screen, (255, 255, 255), (int(eye_x), int(eye_y)), 4)
            pygame.draw.circle(self.screen, (0, 0, 0), (int(eye_x), int(eye_y)), 2)

    def _render_ui(self):
        score_text = f"Score: {self.score:.2f}"
        steps_text = f"Steps: {self.steps}/{self.MAX_STEPS}"

        # Render with shadow for readability
        def render_text_shadow(text, pos, font, color, shadow_color):
            text_surf = font.render(text, True, shadow_color)
            self.screen.blit(text_surf, (pos[0] + 1, pos[1] + 1))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        render_text_shadow(score_text, (10, 35), self.font, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        render_text_shadow(steps_text, (self.WIDTH - 150, 35), self.font, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not run in a headless environment, which is fine
    try:
        os.environ.pop("SDL_VIDEODRIVER")
    except KeyError:
        pass # It was not set, which is fine
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Frogger River Crossing")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    while not terminated and not truncated:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4

        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        obs, reward, term, trunc, info = env.step(action)
        terminated = term
        truncated = trunc
        total_reward += reward
        
        # The environment returns an observation rotated for ML.
        # We need to rotate it back for display.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Steps: {info['steps']}")
            # Reset after a pause
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            terminated = False
            truncated = False

    env.close()