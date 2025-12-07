
# Generated: 2025-08-28T05:02:13.224805
# Source Brief: brief_02499.md
# Brief Index: 2499

        
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
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓ to adjust power, ←→ to aim. Hold Shift + ←→ to add spin. Press Space to bowl."
    )

    game_description = (
        "Side-view arcade bowling. Aim, add power and spin, and bowl to knock down all 10 pins. "
        "Clear a stage by getting 3 consecutive strikes. Clear 3 stages to win!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        try:
            self.font_sm = pygame.font.SysFont("Consolas", 18)
            self.font_md = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_lg = pygame.font.SysFont("Impact", 60)
        except pygame.error:
            self.font_sm = pygame.font.Font(None, 20)
            self.font_md = pygame.font.Font(None, 28)
            self.font_lg = pygame.font.Font(None, 64)

        # --- Colors ---
        self.COLOR_BG = [(20, 30, 40), (40, 20, 30), (30, 40, 20)]
        self.COLOR_LANE = (60, 120, 80)
        self.COLOR_GUTTER = (40, 40, 50)
        self.COLOR_PIN = (230, 50, 50)
        self.COLOR_PIN_HEAD = (240, 100, 100)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_STRIKE = (255, 220, 0)
        self.COLOR_SPARE = (255, 150, 0)
        self.COLOR_AIM = (100, 255, 100, 150)
        self.COLOR_SPIN = (100, 150, 255, 150)

        # --- Game State ---
        self.pins = []
        self.ball = None
        self.particles = []
        self.game_phase = "AIMING" # AIMING, ROLLING, RESULT, STAGE_CLEAR, GAME_OVER
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.termination_reason = ""
        
        self.stage = 1
        self.frame = 1
        self.attempt = 1
        self.consecutive_strikes = 0
        self.frame_scores = []
        self.pins_down_this_frame = [False] * 10
        
        self.total_time_steps = 180 * self.FPS # 3 stages * 60s
        self.time_left = self.total_time_steps

        self._setup_new_frame()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        self.steps += 1
        self.time_left = max(0, self.time_left - 1)
        
        movement, space_pressed, shift_held = action[0], action[1] == 1, action[2] == 1

        if self.game_phase == "AIMING":
            reward += self._handle_aiming(movement, space_pressed, shift_held)
        elif self.game_phase == "ROLLING":
            self._update_physics()
            if self._is_simulation_over():
                reward += self._process_attempt_end()
        elif self.game_phase in ["RESULT", "STAGE_CLEAR"]:
            # A no-op action advances from result screens
            if self.consecutive_strikes >= 3:
                self.stage += 1
                self.consecutive_strikes = 0
                if self.stage > 3:
                    self.game_phase = "GAME_OVER"
                    self.termination_reason = "YOU WIN!"
                    reward += 100
                else:
                    self._setup_new_frame()
                    self.game_phase = "AIMING"
                    reward += 50
            else:
                self._setup_new_frame()
                self.game_phase = "AIMING"

        self._update_particles()
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.termination_reason == "":
                self.termination_reason = "TIME'S UP!"
            self.game_phase = "GAME_OVER"

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_aiming(self, movement, space_pressed, shift_held):
        reward = 0
        if movement == 1: self.aim_power = min(100, self.aim_power + 2)  # Up
        elif movement == 2: self.aim_power = max(0, self.aim_power - 2)   # Down
        elif movement == 3: # Left
            if shift_held: self.aim_spin = max(-100, self.aim_spin - 5)
            else: self.aim_angle = max(-3.0, self.aim_angle - 0.2)
        elif movement == 4: # Right
            if shift_held: self.aim_spin = min(100, self.aim_spin + 5)
            else: self.aim_angle = min(3.0, self.aim_angle + 0.2)

        if space_pressed:
            if self.aim_power < 20 and self.aim_spin == 0:
                reward -= 0.2 # Penalty for "safe" action

            angle_rad = math.radians(self.aim_angle)
            power_scaled = self.aim_power * 0.15
            self.ball['vel'] = [power_scaled * math.cos(angle_rad), power_scaled * math.sin(angle_rad)]
            self.ball['spin'] = self.aim_spin * 0.001
            self.game_phase = "ROLLING"
            # sfx: ball_roll_start
        return reward

    def _update_physics(self):
        # Update ball
        self.ball['pos'][0] += self.ball['vel'][0]
        self.ball['pos'][1] += self.ball['vel'][1]
        self.ball['vel'][1] += self.ball['spin'] # Apply spin
        self.ball['vel'][0] *= 0.995 # Friction
        self.ball['vel'][1] *= 0.995

        # Ball-gutter collision
        if not (240 < self.ball['pos'][1] < 360):
            self.ball['vel'][0] = 0 # Gutter ball
        
        # Update pins
        for pin in self.pins:
            if pin['state'] == 'standing':
                pin['pos'][0] += pin['vel'][0]
                pin['pos'][1] += pin['vel'][1]
                pin['angle'] += pin['ang_vel']
                pin['vel'][0] *= 0.95
                pin['vel'][1] *= 0.95
                pin['ang_vel'] *= 0.95
                if abs(pin['angle']) > 60:
                    pin['state'] = 'down'

        # Collisions
        # Ball-Pin
        for i, pin in enumerate(self.pins):
            if pin['state'] == 'standing':
                dist = math.hypot(self.ball['pos'][0] - pin['pos'][0], self.ball['pos'][1] - pin['pos'][1])
                if dist < self.ball['radius'] + pin['radius']:
                    self._resolve_collision(self.ball, pin)
                    self._create_particles(pin['pos'], self.COLOR_PIN)
                    # sfx: pin_impact

        # Pin-Pin
        for i in range(len(self.pins)):
            for j in range(i + 1, len(self.pins)):
                pin1, pin2 = self.pins[i], self.pins[j]
                if pin1['state'] == 'standing' and pin2['state'] == 'standing':
                    dist = math.hypot(pin1['pos'][0] - pin2['pos'][0], pin1['pos'][1] - pin2['pos'][1])
                    if dist < pin1['radius'] * 2:
                        self._resolve_collision(pin1, pin2)

    def _resolve_collision(self, obj1, obj2):
        dx = obj2['pos'][0] - obj1['pos'][0]
        dy = obj2['pos'][1] - obj1['pos'][1]
        dist = math.hypot(dx, dy)
        if dist == 0: return
        
        nx, ny = dx / dist, dy / dist
        
        v1_n = obj1['vel'][0] * nx + obj1['vel'][1] * ny
        v1_t = -obj1['vel'][0] * ny + obj1['vel'][1] * nx
        v2_n = obj2['vel'][0] * nx + obj2['vel'][1] * ny
        v2_t = -obj2['vel'][0] * ny + obj2['vel'][1] * nx
        
        # Swap normal velocities
        v1_n, v2_n = v2_n, v1_n

        # Convert back
        obj1['vel'][0] = (v1_n * nx - v1_t * ny) * 0.9 # Damping
        obj1['vel'][1] = (v1_n * ny + v1_t * nx) * 0.9
        obj2['vel'][0] = (v2_n * nx - v2_t * ny) * 0.9
        obj2['vel'][1] = (v2_n * ny + v2_t * nx) * 0.9

        # Apply angular velocity to pins
        if 'ang_vel' in obj2:
            obj2['ang_vel'] += (v1_t - v2_t) * 0.1

    def _is_simulation_over(self):
        ball_stopped = math.hypot(*self.ball['vel']) < 0.01
        if self.ball['pos'][0] > self.WIDTH: ball_stopped = True

        pins_stopped = True
        for pin in self.pins:
            if pin['state'] == 'standing' and math.hypot(*pin['vel']) > 0.01:
                pins_stopped = False
                break
        return ball_stopped and pins_stopped
    
    def _process_attempt_end(self):
        reward = 0
        pins_down_this_attempt = 0
        for i, pin in enumerate(self.pins):
            if not self.pins_down_this_frame[i] and pin['state'] == 'down':
                self.pins_down_this_frame[i] = True
                pins_down_this_attempt += 1
        
        reward += pins_down_this_attempt
        
        total_pins_down = sum(self.pins_down_this_frame)
        is_strike = self.attempt == 1 and total_pins_down == 10
        is_spare = self.attempt == 2 and total_pins_down == 10

        self.result_text = ""
        if is_strike:
            reward += 10
            self.consecutive_strikes += 1
            self.result_text = "STRIKE!"
            self.frame_scores.append({'type': 'strike', 'pins': 10})
            self.attempt = 1
            self.frame += 1
            # sfx: strike
        elif is_spare:
            reward += 5
            self.consecutive_strikes = 0
            self.result_text = "SPARE"
            self.frame_scores.append({'type': 'spare', 'pins': total_pins_down})
            self.attempt = 1
            self.frame += 1
            # sfx: spare
        elif self.attempt == 1:
            self.attempt = 2
            self.result_text = f"+{pins_down_this_attempt}"
        elif self.attempt == 2:
            self.consecutive_strikes = 0
            if total_pins_down < 5:
                self.game_over = True
                self.termination_reason = "FAILED TO CLEAR 5 PINS"
            self.frame_scores.append({'type': 'open', 'pins': total_pins_down})
            self.attempt = 1
            self.frame += 1
        
        self.score += reward # Simplified scoring for RL
        self.game_phase = "RESULT"
        if self.consecutive_strikes >= 3:
            self.game_phase = "STAGE_CLEAR"
            self.result_text = f"STAGE {self.stage} CLEAR!"

        return reward

    def _setup_new_frame(self):
        self.game_phase = "AIMING"
        self.attempt = 1
        self.pins_down_this_frame = [False] * 10
        self._reset_entities()
    
    def _reset_entities(self):
        self.ball = {
            'pos': [100, self.HEIGHT / 2 + 50],
            'vel': [0, 0],
            'spin': 0,
            'radius': 12,
            'trail': []
        }
        self.aim_angle = 0
        self.aim_power = 50
        self.aim_spin = 0

        self.pins = []
        pin_layout = [(0, 0), (-1, 1), (1, 1), (-2, 2), (0, 2), (2, 2), (-3, 3), (-1, 3), (1, 3), (3, 3)]
        base_x = self.WIDTH - 120
        base_y = self.HEIGHT / 2 + 50
        dispersion = self.stage * 1.5

        for i, (px, py) in enumerate(pin_layout):
            if not self.pins_down_this_frame[i]:
                self.pins.append({
                    'pos': [base_x + py * 20 + self.np_random.uniform(-dispersion, dispersion),
                            base_y + px * 15 + self.np_random.uniform(-dispersion, dispersion)],
                    'vel': [0, 0],
                    'angle': 0,
                    'ang_vel': 0,
                    'radius': 5,
                    'state': 'standing'
                })

    def _check_termination(self):
        return self.time_left <= 0 or self.stage > 3 or self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG[self.stage - 1])
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "frame": self.frame,
            "attempt": self.attempt,
            "time_left": self.time_left / self.FPS
        }

    def _render_game(self):
        # Lane
        pygame.draw.rect(self.screen, self.COLOR_LANE, (0, 240, self.WIDTH, 120))
        pygame.draw.rect(self.screen, self.COLOR_GUTTER, (0, 220, self.WIDTH, 20))
        pygame.draw.rect(self.screen, self.COLOR_GUTTER, (0, 360, self.WIDTH, 20))

        # Ball trail
        if self.game_phase == "ROLLING":
            self.ball['trail'].append(list(self.ball['pos']))
            if len(self.ball['trail']) > 20:
                self.ball['trail'].pop(0)
        for i, pos in enumerate(self.ball['trail']):
            alpha = int(255 * (i / 20))
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), self.ball['radius'] - (20 - i)//2, (*self.COLOR_BALL, alpha))

        # Pins
        for pin in self.pins:
            if pin['state'] == 'standing':
                self._render_pin(pin)

        # Ball
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball['pos'][0]), int(self.ball['pos'][1]), self.ball['radius'], self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball['pos'][0]), int(self.ball['pos'][1]), self.ball['radius'], self.COLOR_BALL)
        
        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['size']))

    def _render_pin(self, pin):
        x, y = int(pin['pos'][0]), int(pin['pos'][1])
        angle_rad = math.radians(pin['angle'])
        
        body_h, head_h, width = 20, 8, 8
        
        points = [
            (-width/2, body_h/2), (width/2, body_h/2),
            (width/2, -body_h/2), (-width/2, -body_h/2)
        ]
        
        rotated_points = []
        for px, py in points:
            rx = px * math.cos(angle_rad) - py * math.sin(angle_rad)
            ry = px * math.sin(angle_rad) + py * math.cos(angle_rad)
            rotated_points.append((x + rx, y + ry))
            
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_PIN)
        
        # Head
        hx, hy = -width/2, -body_h/2 - head_h/2
        rhx = hx * math.cos(angle_rad) - hy * math.sin(angle_rad)
        rhy = hx * math.sin(angle_rad) + hy * math.cos(angle_rad)
        pygame.gfxdraw.filled_circle(self.screen, int(x+rhx), int(y+rhy), int(width/2), self.COLOR_PIN_HEAD)

    def _render_ui(self):
        # Top bar UI
        self._render_text(f"SCORE: {int(self.score)}", (10, 10), self.font_md, self.COLOR_TEXT, align="left")
        time_str = f"TIME: {int(self.time_left / self.FPS):02d}"
        self._render_text(time_str, (self.WIDTH / 2, 10), self.font_md, self.COLOR_TEXT)
        self._render_text(f"STAGE: {self.stage}/3 | FRAME: {self.frame}", (self.WIDTH - 10, 10), self.font_md, self.COLOR_TEXT, align="right")
        
        # Aiming UI
        if self.game_phase == "AIMING":
            self._render_aim_indicator()
            self._render_text(f"ATTEMPT: {self.attempt}", (self.WIDTH/2, self.HEIGHT-20), self.font_md, self.COLOR_TEXT)

        # Result/Game Over UI
        if self.game_phase in ["RESULT", "STAGE_CLEAR", "GAME_OVER"]:
            color = self.COLOR_STRIKE if "STRIKE" in self.result_text else self.COLOR_SPARE if "SPARE" in self.result_text else self.COLOR_TEXT
            if "CLEAR" in self.result_text: color = self.COLOR_STRIKE
            if "GAME_OVER" in self.game_phase: self.result_text = self.termination_reason
            
            s = pygame.Surface((self.WIDTH, 150), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0, self.HEIGHT/2 - 75))

            self._render_text(self.result_text, (self.WIDTH/2, self.HEIGHT/2), self.font_lg, color)
            if self.game_phase == "GAME_OVER":
                self._render_text(f"Final Score: {int(self.score)}", (self.WIDTH/2, self.HEIGHT/2 + 50), self.font_md, self.COLOR_TEXT)


    def _render_aim_indicator(self):
        start_pos = self.ball['pos']
        length = 30 + self.aim_power * 0.5
        angle_rad = math.radians(self.aim_angle)
        end_pos = (start_pos[0] + length * math.cos(angle_rad), start_pos[1] + length * math.sin(angle_rad))
        pygame.draw.line(self.screen, self.COLOR_AIM, start_pos, end_pos, 3)
        
        # Power bar
        pygame.draw.rect(self.screen, (50,50,50), (20, self.HEIGHT - 40, 100, 20))
        pygame.draw.rect(self.screen, self.COLOR_AIM, (20, self.HEIGHT - 40, self.aim_power, 20))
        self._render_text("POWER", (70, self.HEIGHT-55), self.font_sm, self.COLOR_TEXT)

        # Spin bar
        pygame.draw.rect(self.screen, (50,50,50), (140, self.HEIGHT - 40, 100, 20))
        spin_w = self.aim_spin / 2
        pygame.draw.rect(self.screen, self.COLOR_SPIN, (190 + min(0, spin_w), self.HEIGHT - 40, abs(spin_w), 20))
        self._render_text("SPIN", (190, self.HEIGHT-55), self.font_sm, self.COLOR_TEXT)

    def _render_text(self, text, pos, font, color, align="center"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "center":
            text_rect.center = pos
        elif align == "left":
            text_rect.topleft = pos
        elif align == "right":
            text_rect.topright = pos
        self.screen.blit(text_surface, text_rect)
        
    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'size': self.np_random.uniform(2, 5),
                'lifetime': self.np_random.integers(15, 30),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifetime'] -= 1
            p['size'] *= 0.95
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This part is for human testing and requires a display
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Arcade Bowling")
    except pygame.error:
        print("Pygame display not available. Manual play is disabled.")
        env.close()
        exit()

    obs, info = env.reset()
    terminated = False
    
    # Game loop for manual play
    while not terminated:
        action = [0, 0, 0] # Default no-op action
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # Only step on an event to respect auto_advance=False
        event_occurred = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                event_occurred = True
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
        
        # In manual play, we can choose to advance on key holds or just key presses.
        # Let's advance on any key being held down or if the game is simulating.
        if any(keys) or env.game_phase not in ["AIMING", "GAME_OVER"]:
             event_occurred = True

        if event_occurred and not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render the observation from the environment to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()