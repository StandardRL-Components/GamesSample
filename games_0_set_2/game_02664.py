
# Generated: 2025-08-27T21:03:15.259288
# Source Brief: brief_02664.md
# Brief Index: 2664

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: → to accelerate, ← to brake, ↑↓ to steer. Hold space for a slime boost."
    )

    # Short, user-facing description of the game
    game_description = (
        "Fast-paced arcade snail racer. Manage your slime to boost past opponents and be the first to cross the finish line!"
    )

    # Frames auto-advance for smooth, real-time gameplay
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 2000

    # Colors
    COLOR_BG_SKY = (135, 206, 235)
    COLOR_BG_SKY_DARK = (100, 160, 200)
    COLOR_TRACK = (139, 119, 101)
    COLOR_TRACK_LINE = (120, 100, 80)
    COLOR_FINISH_LIGHT = (255, 255, 255)
    COLOR_FINISH_DARK = (200, 200, 200)
    
    COLOR_PLAYER = (50, 205, 50) # Lime Green
    COLOR_AI_1 = (255, 69, 0)   # Red-Orange
    COLOR_AI_2 = (30, 144, 255) # Dodger Blue
    
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (50, 50, 50)
    COLOR_SLIME_BAR_BG = (70, 70, 70)

    # Track layout
    TRACK_Y_TOP = 100
    TRACK_Y_BOTTOM = 350
    TRACK_MARGIN = 20
    START_X = 50
    FINISH_X = 580

    # Snail Physics
    ACCEL_RATE = 0.15
    BRAKE_RATE = 0.2
    TURN_RATE = 2.5
    FRICTION = 0.98
    BOOST_FORCE = 0.4
    MAX_SPEED = 5.0

    # Slime Mechanics
    SLIME_COST_IDLE = 0.02
    SLIME_COST_MOVE = 0.05
    SLIME_COST_TURN = 0.03
    SLIME_COST_BOOST = 0.25

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font_small = pygame.font.SysFont("Arial", 14, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 24, bold=True)
        
        # Internal state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.snails = []
        self.player_snail = None
        self.previous_player_rank = 0
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        snail_y_positions = np.linspace(
            self.TRACK_Y_TOP + self.TRACK_MARGIN, 
            self.TRACK_Y_BOTTOM - self.TRACK_MARGIN, 
            3
        )
        
        self.player_snail = Snail(
            "Player", self.COLOR_PLAYER, (self.START_X, snail_y_positions[1]), self.np_random
        )
        self.snails = [
            self.player_snail,
            Snail("AI 1", self.COLOR_AI_1, (self.START_X, snail_y_positions[0]), self.np_random),
            Snail("AI 2", self.COLOR_AI_2, (self.START_X, snail_y_positions[2]), self.np_random),
        ]
        
        self._update_race_ranks()
        self.previous_player_rank = self.player_snail.rank

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack action
        movement, space_held, _ = action
        
        # --- Update Game Logic ---
        self._update_player_snail(movement, space_held == 1)
        self._update_ai_snails()

        reward = 0
        
        for snail in self.snails:
            snail.update_physics()
            self._handle_track_boundaries(snail)
            if snail is self.player_snail:
                reward += self._calculate_movement_reward(snail)

        self._update_race_ranks()

        # Overtake reward
        if self.player_snail.rank < self.previous_player_rank:
            reward += 5.0
        self.previous_player_rank = self.player_snail.rank
        
        self.score += reward
        self.steps += 1
        
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.score += terminal_reward
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player_snail(self, movement, is_boosting):
        # movement: 0=none, 1=up, 2=down, 3=left, 4=right
        # Mapping: up/down -> steer, left -> brake, right -> accelerate
        
        # Steering
        if movement == 1: # Up
            self.player_snail.turn(-self.TURN_RATE)
        elif movement == 2: # Down
            self.player_snail.turn(self.TURN_RATE)

        # Acceleration/Braking
        if movement == 4: # Right
            self.player_snail.accelerate(self.ACCEL_RATE)
        elif movement == 3: # Left
            self.player_snail.brake(self.BRAKE_RATE)
        
        # Boosting
        if is_boosting:
            self.player_snail.boost(self.BOOST_FORCE)
            
        self.player_snail.update_slime_cost(movement, is_boosting)

    def _update_ai_snails(self):
        for snail in self.snails:
            if snail is not self.player_snail:
                # Simple AI: try to stay in the middle of their "lane" and accelerate
                target_y = snail.initial_y
                current_y = snail.pos.y
                
                if current_y < target_y - 2:
                    snail.turn(self.TURN_RATE * 0.5)
                elif current_y > target_y + 2:
                    snail.turn(-self.TURN_RATE * 0.5)
                
                # Add some random wobble
                if self.np_random.random() < 0.1:
                    snail.turn(self.np_random.uniform(-1, 1) * self.TURN_RATE)

                snail.accelerate(self.ACCEL_RATE * self.np_random.uniform(0.85, 1.0))
                snail.update_slime_cost(4, False) # Assume they are always accelerating

    def _handle_track_boundaries(self, snail):
        if snail.pos.y < self.TRACK_Y_TOP + snail.RADIUS:
            snail.pos.y = self.TRACK_Y_TOP + snail.RADIUS
            snail.vel.y *= -0.5 # Bounce
            # sound: soft_thud.wav
        elif snail.pos.y > self.TRACK_Y_BOTTOM - snail.RADIUS:
            snail.pos.y = self.TRACK_Y_BOTTOM - snail.RADIUS
            snail.vel.y *= -0.5 # Bounce
            # sound: soft_thud.wav

    def _update_race_ranks(self):
        sorted_snails = sorted(self.snails, key=lambda s: s.pos.x, reverse=True)
        for i, snail in enumerate(sorted_snails):
            snail.rank = i + 1

    def _calculate_movement_reward(self, snail):
        reward = 0
        # Reward for forward velocity
        if snail.vel.x > 0:
            reward += 0.1 * (snail.vel.x / self.MAX_SPEED)
        # Penalty for backward movement
        elif snail.vel.x < 0:
            reward -= 0.2
        # Penalty for slime usage
        reward -= snail.last_slime_cost * 0.01
        return reward

    def _check_termination(self):
        # Max steps
        if self.steps >= self.MAX_STEPS:
            return True, 0

        # Slime out
        for snail in self.snails:
            if snail.slime <= 0:
                reward = -50.0 if snail is self.player_snail else 0
                return True, reward
        
        # Finish line
        winner = None
        for snail in self.snails:
            if snail.pos.x > self.FINISH_X:
                if winner is None or snail.pos.x > winner.pos.x:
                    winner = snail

        if winner:
            if winner is self.player_snail:
                # sound: win_fanfare.wav
                return True, 100.0 # Player wins
            else:
                # sound: lose_sound.wav
                return True, -25.0 # AI wins
        
        return False, 0

    def _get_observation(self):
        # --- Render Background ---
        self.screen.fill(self.COLOR_BG_SKY)
        pygame.draw.rect(self.screen, self.COLOR_BG_SKY_DARK, (0, self.HEIGHT/2, self.WIDTH, self.HEIGHT/2))
        
        # --- Render Track ---
        pygame.draw.rect(self.screen, self.COLOR_TRACK, (0, self.TRACK_Y_TOP, self.WIDTH, self.TRACK_Y_BOTTOM - self.TRACK_Y_TOP))
        for i in range(self.TRACK_Y_TOP, self.TRACK_Y_BOTTOM, 10):
            pygame.draw.line(self.screen, self.COLOR_TRACK_LINE, (0, i), (self.WIDTH, i))

        # --- Render Finish Line ---
        for i in range(int(self.TRACK_Y_TOP), int(self.TRACK_Y_BOTTOM), 10):
            for j in range(int(self.FINISH_X), int(self.FINISH_X + 20), 10):
                color = self.COLOR_FINISH_LIGHT if ((i // 10) % 2 == (j // 10) % 2) else self.COLOR_FINISH_DARK
                pygame.draw.rect(self.screen, color, (j, i, 10, 10))
        
        # --- Render Trails (on a separate surface for alpha) ---
        trail_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        for snail in self.snails:
            snail.render_trail(trail_surface)
        self.screen.blit(trail_surface, (0,0))
        
        # --- Render Snails and UI ---
        sorted_by_y = sorted(self.snails, key=lambda s: s.pos.y)
        for snail in sorted_by_y:
            snail.render(self.screen, self.font_small, self)
            snail.render_ui(self.screen, self.font_small, self.font_large)

        if self.game_over:
            self._render_game_over_text()
            
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game_over_text(self):
        winner = next((s for s in self.snails if s.rank == 1 and s.pos.x > self.FINISH_X), None)
        player_lost = self.player_snail.slime <= 0
        
        if winner:
            text = "YOU WIN!" if winner is self.player_snail else f"{winner.name.upper()} WINS!"
            color = self.COLOR_PLAYER if winner is self.player_snail else winner.color
        elif player_lost:
            text = "OUT OF SLIME!"
            color = (200, 0, 0)
        else: # Time out or other loss
            text = "RACE OVER"
            color = self.COLOR_TEXT

        text_surf = self.font_large.render(text, True, color)
        shadow_surf = self.font_large.render(text, True, self.COLOR_TEXT_SHADOW)
        text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_slime": self.player_snail.slime,
            "player_rank": self.player_snail.rank,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

class Snail:
    RADIUS = 12
    SHELL_RADIUS = 10
    
    def __init__(self, name, color, start_pos, np_random):
        self.name = name
        self.color = color
        self.initial_y = start_pos[1]
        self.pos = pygame.math.Vector2(start_pos)
        self.vel = pygame.math.Vector2(0, 0)
        self.angle = 0 # degrees
        self.slime = 100.0
        self.rank = 0
        self.trail = deque(maxlen=60)
        self.last_slime_cost = 0
        self.bob_offset = 0
        self.bob_speed = np_random.uniform(0.1, 0.2)
        self.np_random = np_random

    def accelerate(self, rate):
        accel = pygame.math.Vector2(rate, 0).rotate(-self.angle)
        self.vel += accel
        # sound: engine_accel.wav

    def brake(self, rate):
        if self.vel.length() > 0:
            brake_vec = -self.vel.normalize() * rate
            self.vel += brake_vec
        # sound: brake_squeal.wav

    def turn(self, angle_change):
        self.angle = (self.angle + angle_change) % 360

    def boost(self, force):
        if self.slime > GameEnv.SLIME_COST_BOOST:
            boost_vec = pygame.math.Vector2(force, 0).rotate(-self.angle)
            self.vel += boost_vec
            # sound: boost.wav

    def update_slime_cost(self, movement_action, is_boosting):
        cost = GameEnv.SLIME_COST_IDLE
        if movement_action in [1, 2]: # Turning
            cost += GameEnv.SLIME_COST_TURN
        if movement_action == 4: # Accelerating
            cost += GameEnv.SLIME_COST_MOVE
        if is_boosting and self.slime > GameEnv.SLIME_COST_BOOST:
            cost += GameEnv.SLIME_COST_BOOST
        
        self.slime = max(0, self.slime - cost)
        self.last_slime_cost = cost

    def update_physics(self):
        self.vel *= GameEnv.FRICTION
        if self.vel.length() > GameEnv.MAX_SPEED:
            self.vel.scale_to_length(GameEnv.MAX_SPEED)
        
        self.pos += self.vel
        self.trail.append(self.pos.copy())
        
        self.bob_offset = math.sin(pygame.time.get_ticks() * self.bob_speed * 0.1) * 2

    def render(self, surface, font, env):
        # Speed lines
        if self.vel.length() > GameEnv.MAX_SPEED * 0.7:
            for _ in range(3):
                start_pos = self.pos + pygame.math.Vector2(self.np_random.uniform(-15, 15), self.np_random.uniform(-15, 15))
                end_pos = start_pos - self.vel.normalize() * self.np_random.uniform(5, 10)
                pygame.draw.line(surface, (255,255,255), start_pos, end_pos, 1)

        # Body
        draw_pos = (int(self.pos.x), int(self.pos.y + self.bob_offset))
        pygame.gfxdraw.filled_circle(surface, draw_pos[0], draw_pos[1], self.RADIUS, self.color)
        pygame.gfxdraw.aacircle(surface, draw_pos[0], draw_pos[1], self.RADIUS, (0,0,0,100))

        # Shell with swirl
        shell_color = tuple(max(0, c-40) for c in self.color)
        pygame.gfxdraw.filled_circle(surface, draw_pos[0], draw_pos[1], self.SHELL_RADIUS, shell_color)
        for i in range(5):
            arc_radius = self.SHELL_RADIUS - i*2
            if arc_radius > 0:
                start_angle = (pygame.time.get_ticks() * 0.1 + i * 72) % 360
                end_angle = (start_angle + 180) % 360
                if start_angle > end_angle: start_angle, end_angle = end_angle, start_angle
                pygame.gfxdraw.arc(surface, draw_pos[0], draw_pos[1], arc_radius, int(start_angle), int(end_angle), (0,0,0,50))

        # Eyes
        eye_angle_rad = math.radians(-self.angle)
        eye_base_offset = pygame.math.Vector2(self.RADIUS * 0.6, 0).rotate(-self.angle)
        eye1_pos = self.pos + eye_base_offset.rotate(30) + (0, self.bob_offset)
        eye2_pos = self.pos + eye_base_offset.rotate(-30) + (0, self.bob_offset)
        
        for ep in [eye1_pos, eye2_pos]:
            pygame.draw.circle(surface, (255,255,255), (int(ep.x), int(ep.y)), 4)
            pupil_offset = self.vel.normalize() * 1.5 if self.vel.length() > 0.1 else pygame.math.Vector2(1,0).rotate(-self.angle)*1.5
            pupil_pos = ep + pupil_offset
            pygame.draw.circle(surface, (0,0,0), (int(pupil_pos.x), int(pupil_pos.y)), 2)

    def render_trail(self, surface):
        if len(self.trail) > 2:
            alpha_color = self.color + (100,)
            pygame.draw.lines(surface, alpha_color, False, [tuple(p) for p in self.trail], 5)

    def render_ui(self, surface, small_font, large_font):
        # Slime bar
        bar_width = 40
        bar_height = 6
        bar_x = self.pos.x - bar_width / 2
        bar_y = self.pos.y - self.RADIUS - 15
        
        pygame.draw.rect(surface, GameEnv.COLOR_SLIME_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        fill_width = bar_width * (self.slime / 100.0)
        pygame.draw.rect(surface, self.color, (bar_x, bar_y, fill_width, bar_height))

        # Rank text
        rank_map = {1: "1st", 2: "2nd", 3: "3rd"}
        rank_text = rank_map.get(self.rank, "")
        text_surf = small_font.render(rank_text, True, self.color)
        shadow_surf = small_font.render(rank_text, True, GameEnv.COLOR_TEXT_SHADOW)
        text_rect = text_surf.get_rect(center=(self.pos.x, self.pos.y + self.RADIUS + 10))
        surface.blit(shadow_surf, (text_rect.x+1, text_rect.y+1))
        surface.blit(text_surf, text_rect)