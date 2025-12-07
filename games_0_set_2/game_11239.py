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
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A futuristic 1v1 soccer game where you can slow down time to outmaneuver your opponent."
    )
    user_guide = (
        "Use arrow keys to move. Press space to kick the ball and shift to activate your time-slowing boots."
    )
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_FIELD = (0, 100, 50)
    COLOR_LINES = (0, 200, 100)
    COLOR_PLAYER = (0, 160, 255)
    COLOR_PLAYER_GLOW = (100, 200, 255)
    COLOR_OPPONENT = (255, 64, 64)
    COLOR_OPPONENT_GLOW = (255, 128, 128)
    COLOR_BALL = (255, 255, 255)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TIME_SLOW_OVERLAY = (50, 255, 150, 40) # RGBA

    # Screen and Field Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FIELD_MARGIN = 20
    GOAL_WIDTH = 120
    GOAL_DEPTH = 15

    # Game Physics & Parameters
    PLAYER_ACCEL = 0.8
    PLAYER_DAMPING = 0.92
    PLAYER_RADIUS = 15
    BALL_RADIUS = 8
    BALL_DAMPING = 0.985
    KICK_STRENGTH = 12
    MAX_VEL = 15
    MAX_STEPS = 1800 # 30 seconds at 60 FPS
    
    # AI Parameters
    INITIAL_OPPONENT_SPEED = 1.8
    OPPONENT_SPEED_INCREASE_INTERVAL = 200
    OPPONENT_SPEED_INCREASE_AMOUNT = 0.05
    OPPONENT_MAX_SPEED_FACTOR = 2.0
    OPPONENT_RADIUS = 15

    # Boot Tiers (Unlockable)
    BOOT_SPECS = {
        0: {"name": "Standard", "duration": 30, "cooldown": 240}, # 0.5s duration, 4s cooldown
        1: {"name": "Advanced", "duration": 60, "cooldown": 210}, # 1.0s duration, 3.5s cooldown
        2: {"name": "Chrono", "duration": 90, "cooldown": 180},   # 1.5s duration, 3s cooldown
    }
    WINS_TO_UPGRADE = [3, 6] # Wins needed to reach level 1, 2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        # Persistent state (survives resets)
        self.wins = 0
        self.current_boot_level = 0
        
        # State variables (reset each episode)
        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.last_move_dir = pygame.math.Vector2(1, 0)
        self.opponent_pos = pygame.math.Vector2(0, 0)
        self.opponent_vel = pygame.math.Vector2(0, 0)
        self.opponent_state = "CHASE"
        self.opponent_speed = self.INITIAL_OPPONENT_SPEED
        self.ball_pos = pygame.math.Vector2(0, 0)
        self.ball_vel = pygame.math.Vector2(0, 0)
        self.particles = []
        self.steps = 0
        self.score_player = 0
        self.score_opponent = 0
        self.time_slow_timer = 0
        self.time_slow_cooldown = 0
        self.ball_stuck_timer = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.last_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score_player = 0
        self.score_opponent = 0
        self.game_over = False
        self.last_reward = 0.0

        self._update_boot_level()
        self._reset_positions()

        self.player_vel.update(0, 0)
        self.opponent_vel.update(0, 0)
        self.ball_vel.update(0, 0)
        self.opponent_state = "CHASE"
        self.opponent_speed = self.INITIAL_OPPONENT_SPEED

        self.particles = []
        self.time_slow_timer = 0
        self.time_slow_cooldown = 0
        self.ball_stuck_timer = 0
        self.last_space_held = False
        self.last_shift_held = False

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        reward = 0
        
        # --- Time and Timers ---
        self.steps += 1
        time_factor = 0.5 if self.time_slow_timer > 0 else 1.0
        if self.time_slow_timer > 0: self.time_slow_timer -= 1
        if self.time_slow_cooldown > 0: self.time_slow_cooldown -= 1

        # --- Action Processing ---
        # Movement
        accel = pygame.math.Vector2(0, 0)
        if movement == 1: accel.y = -1
        elif movement == 2: accel.y = 1
        elif movement == 3: accel.x = -1
        elif movement == 4: accel.x = 1
        if accel.length() > 0:
            accel.normalize_ip()
            self.last_move_dir = pygame.math.Vector2(accel)
            # Reward for moving towards ball
            dist_before = self.player_pos.distance_to(self.ball_pos)
            dist_after = (self.player_pos + accel).distance_to(self.ball_pos)
            if dist_after < dist_before:
                reward += 0.001 # Small reward for closing distance
        self.player_vel += accel * self.PLAYER_ACCEL
        
        # Kick (on press)
        if space_held and not self.last_space_held:
            if self.player_pos.distance_to(self.ball_pos) < self.PLAYER_RADIUS + self.BALL_RADIUS + 5:
                # SFX: KICK
                kick_dir = pygame.math.Vector2(self.last_move_dir)
                self.ball_vel += kick_dir * self.KICK_STRENGTH
                self._create_particles(self.ball_pos, 20, self.COLOR_BALL)
                
                # Reward for kicking towards opponent goal
                if self.ball_vel.x > 0:
                    reward += 0.2

        # Time Slow (on press)
        if shift_held and not self.last_shift_held and self.time_slow_cooldown == 0:
            # SFX: TIME_SLOW_ACTIVATE
            boot_spec = self.BOOT_SPECS[self.current_boot_level]
            self.time_slow_timer = boot_spec["duration"]
            self.time_slow_cooldown = boot_spec["cooldown"]

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- Game Logic Updates ---
        self._update_player()
        self._update_opponent(time_factor)
        self._update_ball(time_factor)
        self._update_particles()
        
        # --- Goal Check ---
        goal_reward = self._check_goals()
        reward += goal_reward

        # --- Termination Check ---
        terminated = self.steps >= self.MAX_STEPS
        truncated = False
        if terminated:
            if self.score_player > self.score_opponent:
                reward += 50 # Win
                self.wins += 1
            elif self.score_opponent > self.score_player:
                reward -= 25 # Loss
            # Draw is 0
        
        self.last_reward = reward
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_player(self):
        # Apply damping and limit velocity
        self.player_vel *= self.PLAYER_DAMPING
        if self.player_vel.length() > self.MAX_VEL:
            self.player_vel.scale_to_length(self.MAX_VEL)
        self.player_pos += self.player_vel

        # Boundary collision
        self.player_pos.x = np.clip(self.player_pos.x, self.FIELD_MARGIN, self.SCREEN_WIDTH - self.FIELD_MARGIN)
        self.player_pos.y = np.clip(self.player_pos.y, self.FIELD_MARGIN, self.SCREEN_HEIGHT - self.FIELD_MARGIN)

    def _update_opponent(self, time_factor):
        # Difficulty scaling
        if self.steps % self.OPPONENT_SPEED_INCREASE_INTERVAL == 0 and self.steps > 0:
            max_speed = self.INITIAL_OPPONENT_SPEED * self.OPPONENT_MAX_SPEED_FACTOR
            self.opponent_speed = min(max_speed, self.opponent_speed + self.OPPONENT_SPEED_INCREASE_AMOUNT)

        # State machine AI
        dist_to_ball = self.opponent_pos.distance_to(self.ball_pos)
        
        if self.opponent_state == "CHASE":
            if dist_to_ball < self.OPPONENT_RADIUS + self.BALL_RADIUS + 5:
                self.opponent_state = "KICK"
            else:
                direction = (self.ball_pos - self.opponent_pos).normalize()
                self.opponent_vel = direction * self.opponent_speed
        
        elif self.opponent_state == "KICK":
            # SFX: OPPONENT_KICK
            kick_dir = (pygame.math.Vector2(self.FIELD_MARGIN, self.SCREEN_HEIGHT / 2) - self.opponent_pos).normalize()
            self.ball_vel += kick_dir * self.KICK_STRENGTH * 0.8 # Slightly weaker kick
            self._create_particles(self.ball_pos, 15, self.COLOR_OPPONENT)
            self.opponent_state = "RETURN"
            self.opponent_vel *= 0.1

        elif self.opponent_state == "RETURN":
            return_pos = pygame.math.Vector2(self.SCREEN_WIDTH * 0.75, self.SCREEN_HEIGHT / 2)
            if self.opponent_pos.distance_to(return_pos) < 20:
                self.opponent_state = "CHASE"
            else:
                direction = (return_pos - self.opponent_pos).normalize()
                self.opponent_vel = direction * self.opponent_speed * 0.7

        self.opponent_pos += self.opponent_vel * time_factor
        self.opponent_pos.x = np.clip(self.opponent_pos.x, self.FIELD_MARGIN, self.SCREEN_WIDTH - self.FIELD_MARGIN)
        self.opponent_pos.y = np.clip(self.opponent_pos.y, self.FIELD_MARGIN, self.SCREEN_HEIGHT - self.FIELD_MARGIN)

    def _update_ball(self, time_factor):
        self.ball_vel *= self.BALL_DAMPING
        if self.ball_vel.length() < 0.1:
            self.ball_vel.update(0, 0)
        self.ball_pos += self.ball_vel * time_factor

        # Wall bounces
        if self.ball_pos.y < self.FIELD_MARGIN + self.BALL_RADIUS or self.ball_pos.y > self.SCREEN_HEIGHT - self.FIELD_MARGIN - self.BALL_RADIUS:
            self.ball_vel.y *= -0.9
            self.ball_pos.y = np.clip(self.ball_pos.y, self.FIELD_MARGIN + self.BALL_RADIUS, self.SCREEN_HEIGHT - self.FIELD_MARGIN - self.BALL_RADIUS)
        
        # Anti-softlock
        if self.ball_vel.length() < 0.5 and (self.ball_pos.x < self.FIELD_MARGIN + 5 or self.ball_pos.x > self.SCREEN_WIDTH - self.FIELD_MARGIN - 5):
            self.ball_stuck_timer += 1
        else:
            self.ball_stuck_timer = 0
        
        if self.ball_stuck_timer > 100:
            self._reset_positions(reset_player=False)
            self.ball_stuck_timer = 0


    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _check_goals(self):
        goal_top = (self.SCREEN_HEIGHT - self.GOAL_WIDTH) / 2
        goal_bottom = (self.SCREEN_HEIGHT + self.GOAL_WIDTH) / 2
        reward = 0

        # Opponent goal (Player scores)
        if self.ball_pos.x > self.SCREEN_WIDTH - self.FIELD_MARGIN and goal_top < self.ball_pos.y < goal_bottom:
            # SFX: GOAL_PLAYER
            self.score_player += 1
            reward = 10
            self._create_particles(self.ball_pos, 100, self.COLOR_PLAYER, 5)
            self._reset_positions()
        
        # Player goal (Opponent scores)
        if self.ball_pos.x < self.FIELD_MARGIN and goal_top < self.ball_pos.y < goal_bottom:
            # SFX: GOAL_OPPONENT
            self.score_opponent += 1
            reward = -5
            self._create_particles(self.ball_pos, 100, self.COLOR_OPPONENT, 5)
            self._reset_positions()
        
        return reward

    def _reset_positions(self, reset_player=True):
        if reset_player:
            self.player_pos.update(self.SCREEN_WIDTH * 0.25, self.SCREEN_HEIGHT / 2)
            self.player_vel.update(0, 0)
        self.opponent_pos.update(self.SCREEN_WIDTH * 0.75, self.SCREEN_HEIGHT / 2)
        self.opponent_vel.update(0, 0)
        self.ball_pos.update(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.ball_vel.update(0, 0)

    def _update_boot_level(self):
        if self.wins >= self.WINS_TO_UPGRADE[1]:
            self.current_boot_level = 2
        elif self.wins >= self.WINS_TO_UPGRADE[0]:
            self.current_boot_level = 1
        else:
            self.current_boot_level = 0
            
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Field
        field_rect = pygame.Rect(self.FIELD_MARGIN, self.FIELD_MARGIN, self.SCREEN_WIDTH - 2 * self.FIELD_MARGIN, self.SCREEN_HEIGHT - 2 * self.FIELD_MARGIN)
        pygame.draw.rect(self.screen, self.COLOR_FIELD, field_rect)
        pygame.draw.rect(self.screen, self.COLOR_LINES, field_rect, 2)
        
        # Center line and circle
        pygame.draw.line(self.screen, self.COLOR_LINES, (self.SCREEN_WIDTH/2, self.FIELD_MARGIN), (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT - self.FIELD_MARGIN), 2)
        pygame.gfxdraw.aacircle(self.screen, int(self.SCREEN_WIDTH/2), int(self.SCREEN_HEIGHT/2), 50, self.COLOR_LINES)

        # Goals
        goal_top = (self.SCREEN_HEIGHT - self.GOAL_WIDTH) / 2
        player_goal_rect = pygame.Rect(self.FIELD_MARGIN - self.GOAL_DEPTH, goal_top, self.GOAL_DEPTH, self.GOAL_WIDTH)
        opponent_goal_rect = pygame.Rect(self.SCREEN_WIDTH - self.FIELD_MARGIN, goal_top, self.GOAL_DEPTH, self.GOAL_WIDTH)
        pygame.draw.rect(self.screen, self.COLOR_BG, player_goal_rect)
        pygame.draw.rect(self.screen, self.COLOR_BG, opponent_goal_rect)
        
        # Particles
        for p in self.particles:
            alpha = p['life'] / p['max_life']
            color = (p['color'][0], p['color'][1], p['color'][2], int(alpha * 255))
            size = int(p['radius'] * alpha)
            if size > 0:
                self._draw_particle_circle(p['pos'], size, color)

        # Ball Trail (from recent velocity)
        if self.ball_vel.length() > 1:
            for i in range(5):
                t = i / 5.0
                pos = self.ball_pos - self.ball_vel * t * 0.3
                radius = self.BALL_RADIUS * (1 - t) * 0.8
                alpha = 200 * (1 - t)
                if radius > 1:
                    self._draw_particle_circle(pos, int(radius), (self.COLOR_BALL[0], self.COLOR_BALL[1], self.COLOR_BALL[2], int(alpha)))
        
        # Entities
        self._draw_entity_with_glow(self.player_pos, self.PLAYER_RADIUS, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)
        self._draw_entity_with_glow(self.opponent_pos, self.OPPONENT_RADIUS, self.COLOR_OPPONENT, self.COLOR_OPPONENT_GLOW)
        self._draw_entity_with_glow(self.ball_pos, self.BALL_RADIUS, self.COLOR_BALL, self.COLOR_BALL)
        
        # Time slow overlay
        if self.time_slow_timer > 0:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_TIME_SLOW_OVERLAY)
            self.screen.blit(overlay, (0, 0))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"{self.score_player}", True, self.COLOR_PLAYER)
        self.screen.blit(score_text, (40, 10))
        score_text_opp = self.font_main.render(f"{self.score_opponent}", True, self.COLOR_OPPONENT)
        score_rect_opp = score_text_opp.get_rect(right=self.SCREEN_WIDTH - 40, top=10)
        self.screen.blit(score_text_opp, score_rect_opp)
        
        # Timer
        time_left = (self.MAX_STEPS - self.steps) / 60
        timer_text = self.font_main.render(f"{time_left:.1f}", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(centerx=self.SCREEN_WIDTH / 2, top=10)
        self.screen.blit(timer_text, timer_rect)
        
        # Boot Icon
        boot_spec = self.BOOT_SPECS[self.current_boot_level]
        boot_text = self.font_small.render(f"Boot: {boot_spec['name']}", True, self.COLOR_TEXT)
        self.screen.blit(boot_text, (20, self.SCREEN_HEIGHT - 30))
        
        # Time Slow Cooldown Bar
        cooldown_pct = self.time_slow_cooldown / boot_spec["cooldown"] if boot_spec["cooldown"] > 0 else 0
        bar_width = 100
        bar_height = 10
        bar_x = 20
        bar_y = self.SCREEN_HEIGHT - 50
        pygame.draw.rect(self.screen, (50,50,50), (bar_x, bar_y, bar_width, bar_height))
        fill_width = bar_width * (1 - cooldown_pct)
        fill_color = self.COLOR_TIME_SLOW_OVERLAY[:3] if cooldown_pct == 0 else (100,120,110)
        pygame.draw.rect(self.screen, fill_color, (bar_x, bar_y, fill_width, bar_height))

    def _draw_entity_with_glow(self, pos, radius, color, glow_color):
        int_pos = (int(pos.x), int(pos.y))
        # Glow effect
        for i in range(4):
            alpha = 80 - i * 20
            current_radius = radius + i * 2
            pygame.gfxdraw.filled_circle(self.screen, int_pos[0], int_pos[1], current_radius, (*glow_color, alpha))
        
        # Main body
        pygame.gfxdraw.aacircle(self.screen, int_pos[0], int_pos[1], radius, color)
        pygame.gfxdraw.filled_circle(self.screen, int_pos[0], int_pos[1], radius, color)

    def _draw_particle_circle(self, pos, radius, color_rgba):
        surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, color_rgba, (radius, radius), radius)
        self.screen.blit(surf, (int(pos.x - radius), int(pos.y - radius)), special_flags=pygame.BLEND_RGBA_ADD)

    def _create_particles(self, pos, count, color, speed=2):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            vel_mag = random.uniform(0.5, 1.5) * speed
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * vel_mag
            self.particles.append({
                'pos': pygame.math.Vector2(pos),
                'vel': vel,
                'life': random.randint(20, 40),
                'max_life': 40,
                'color': color,
                'radius': random.uniform(2, 5)
            })

    def _get_info(self):
        return {
            "score_player": self.score_player,
            "score_opponent": self.score_opponent,
            "steps": self.steps,
            "wins": self.wins,
            "boot_level": self.current_boot_level,
            "last_reward": self.last_reward
        }
    
    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # Example usage: Play the game with random actions or keyboard control
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Keyboard Control Setup ---
    # To play manually, uncomment this section and run the script.
    # Use arrow keys for movement, Space to kick, Left Shift to slow time.
    
    # done = False
    # clock = pygame.time.Clock()
    # display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    # pygame.display.set_caption("Soccer Time-Slow")

    # while not done:
    #     movement = 0 # no-op
    #     space = 0
    #     shift = 0

    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_UP]: movement = 1
    #     elif keys[pygame.K_DOWN]: movement = 2
    #     elif keys[pygame.K_LEFT]: movement = 3
    #     elif keys[pygame.K_RIGHT]: movement = 4
        
    #     if keys[pygame.K_SPACE]: space = 1
    #     if keys[pygame.K_LSHIFT]: shift = 1
        
    #     action = [movement, space, shift]
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated

    #     # Render to the display window
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     display_screen.blit(surf, (0, 0))
    #     pygame.display.flip()

    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             done = True
        
    #     clock.tick(60) # Limit to 60 FPS
    
    # env.close()
    
    # --- Random Agent Test ---
    print("Running random agent test for 1000 steps...")
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if (i+1) % 100 == 0:
            print(f"Step {i+1}: Reward={reward:.2f}, Info={info}")
        if terminated or truncated:
            print("Episode finished. Resetting.")
            obs, info = env.reset()
    env.close()
    print("Random agent test complete.")