import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:50:51.653408
# Source Brief: brief_00732.md
# Brief Index: 732
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Shoot an enchanted ball into the opponent's goal in this futuristic, gravity-defying sports game. "
        "Time your shots and cycle through ball enchantments to outwit your defenders."
    )
    user_guide = (
        "Controls: ←→ to move. Press space to shoot the ball. Press shift to cycle through ball enchantments."
    )
    auto_advance = True

    # Class attribute for persistent unlocks across resets
    unlocked_enchantments = ['standard']
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1500 # Increased from 1000 to allow for more gameplay
        self.WIN_SCORE = 10
        self.LOSE_SCORE = 5
        
        # --- Colors ---
        self.COLOR_BG = (15, 10, 40)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 255, 50)
        self.COLOR_OPPONENT = (150, 255, 150)
        self.COLOR_OPPONENT_GLOW = (150, 255, 150, 40)
        self.COLOR_COURT = (200, 200, 255, 150)
        self.COLOR_GOAL = (255, 200, 0)
        self.COLOR_TEXT = (240, 240, 255)
        self.BALL_COLORS = {
            'standard': (255, 120, 0),
            'gravity': (200, 0, 255),
            'speed': (255, 50, 50)
        }
        
        # --- Physics & Gameplay ---
        self.PLAYER_SPEED = 8
        self.BALL_RADIUS = 10
        self.PLAYER_RADIUS = 15
        self.OPPONENT_RADIUS = 18
        self.GRAVITY = 0.5
        self.BOUNCE_FACTOR = -0.8
        self.SHOOT_POWER = 15
        self.AIM_OSCILLATION_SPEED = 0.05
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        # --- State Variables ---
        # These are initialized properly in reset()
        self.steps = 0
        self.score = 0
        self.opponent_score = 0
        self.game_over = False
        
        self.player_pos = np.array([0.0, 0.0])
        self.ball = {}
        self.opponents = []
        self.particles = []
        
        self.aim_angle = 0.0
        self.current_enchantment_idx = 0
        self.shift_was_held = False
        self.opponent_base_speed = 2.0
        self.opponent_current_speed = 2.0
        
        self.reward_this_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.opponent_score = 0
        self.game_over = False
        
        self.player_pos = np.array([self.WIDTH * 0.75, self.HEIGHT - 40.0])
        
        self.ball = self._create_ball(on_player=True)

        self.opponents = [
            {'pos': np.array([self.WIDTH * 0.25, self.HEIGHT / 3]), 'dir': 1, 'pattern': 'vertical'},
            {'pos': np.array([self.WIDTH * 0.5, self.HEIGHT / 2]), 'dir': 1, 'pattern': 'sinusoidal', 'offset': 0}
        ]
        
        self.particles = []
        self.aim_angle = -math.pi / 4  # Start at -45 degrees
        self.current_enchantment_idx = 0
        self.shift_was_held = False
        self.opponent_current_speed = self.opponent_base_speed
        
        # Add some background stars for visual interest
        self.stars = [
            (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT), random.randint(1, 2))
            for _ in range(100)
        ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.reward_this_step = 0
        self.steps += 1
        
        prev_player_ball_dist = np.linalg.norm(self.player_pos - self.ball['pos']) if not self.ball['on_player'] else 0

        self._handle_input(action)
        self._update_game_state()
        
        # --- Continuous Reward ---
        if not self.ball['on_player']:
            current_player_ball_dist = np.linalg.norm(self.player_pos - self.ball['pos'])
            if current_player_ball_dist < prev_player_ball_dist:
                self.reward_this_step += 0.1 # Moved towards ball
            else:
                self.reward_this_step -= 0.1 # Moved away from ball

        terminated = self._check_termination()

        # Add terminal rewards
        if terminated:
            if self.score >= self.WIN_SCORE:
                self.reward_this_step += 100
            if self.opponent_score >= self.LOSE_SCORE:
                self.reward_this_step -= 100
        
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _create_ball(self, on_player=False):
        ball_type = self.unlocked_enchantments[self.current_enchantment_idx]
        if on_player:
            pos = self.player_pos + np.array([0, -self.PLAYER_RADIUS - self.BALL_RADIUS - 5])
            vel = np.array([0.0, 0.0])
        else: # Reset to center
            pos = np.array([self.WIDTH / 2, self.HEIGHT / 2])
            vel = np.array([0.0, 0.0])
            
        return {
            'pos': pos, 'vel': vel, 'on_player': on_player, 'type': ball_type,
            'gravity_mod': 1, 'stuck_timer': 0
        }

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Movement ---
        if movement == 3: # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos[0] += self.PLAYER_SPEED
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)

        # --- Shooting ---
        if space_held and self.ball['on_player']:
            # sfx: shoot_sound()
            ball_type = self.ball['type']
            power = self.SHOOT_POWER * 1.5 if ball_type == 'speed' else self.SHOOT_POWER
            self.ball['vel'] = np.array([math.cos(self.aim_angle) * power, math.sin(self.aim_angle) * power])
            self.ball['on_player'] = False
            if ball_type == 'gravity':
                self.ball['gravity_mod'] = 1

        # --- Cycle Enchantment ---
        if shift_held and not self.shift_was_held:
            # sfx: cycle_enchantment_sound()
            self.current_enchantment_idx = (self.current_enchantment_idx + 1) % len(self.unlocked_enchantments)
            self.ball['type'] = self.unlocked_enchantments[self.current_enchantment_idx]
        self.shift_was_held = shift_held

    def _update_game_state(self):
        # --- Update Aim Angle ---
        self.aim_angle += self.AIM_OSCILLATION_SPEED
        if self.aim_angle > -math.pi / 6 or self.aim_angle < -5 * math.pi / 6:
            self.AIM_OSCILLATION_SPEED *= -1
            self.aim_angle = np.clip(self.aim_angle, -5 * math.pi / 6, -math.pi / 6)

        # --- Update Opponents ---
        self.opponent_current_speed = self.opponent_base_speed + 0.05 * (self.steps // 500)
        for op in self.opponents:
            if op['pattern'] == 'vertical':
                op['pos'][1] += self.opponent_current_speed * op['dir']
                if op['pos'][1] < self.OPPONENT_RADIUS or op['pos'][1] > self.HEIGHT - self.OPPONENT_RADIUS:
                    op['dir'] *= -1
            elif op['pattern'] == 'sinusoidal':
                op['offset'] += 0.03
                op['pos'][0] = self.WIDTH * 0.25 + math.sin(op['offset']) * (self.WIDTH * 0.2)
        
        # --- Update Ball ---
        if self.ball['on_player']:
            self.ball['pos'] = self.player_pos + np.array([0, -self.PLAYER_RADIUS - self.BALL_RADIUS - 2])
        else:
            self.ball['vel'][1] += self.GRAVITY * self.ball.get('gravity_mod', 1)
            self.ball['pos'] += self.ball['vel']
            
            # Anti-softlock
            if np.linalg.norm(self.ball['vel']) < 0.1:
                self.ball['stuck_timer'] += 1
            else:
                self.ball['stuck_timer'] = 0
            
            if self.ball['stuck_timer'] > self.FPS * 2:
                self.ball = self._create_ball(on_player=True) # Reset to player
                self.reward_this_step -= 1 # Penalty for stuck ball

        # --- Ball-Wall Collisions ---
        if self.ball['pos'][1] < self.BALL_RADIUS or self.ball['pos'][1] > self.HEIGHT - self.BALL_RADIUS:
            self.ball['pos'][1] = np.clip(self.ball['pos'][1], self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS)
            self.ball['vel'][1] *= self.BOUNCE_FACTOR
            if self.ball['type'] == 'gravity':
                self.ball['gravity_mod'] *= -1
            # sfx: bounce_sound()

        # --- Scoring ---
        goal_height = self.HEIGHT / 2
        if self.ball['pos'][0] < self.BALL_RADIUS and self.ball['pos'][1] > self.HEIGHT - goal_height: # Opponent Goal
            self.score += 1
            self.reward_this_step += 5
            self._spawn_particles(self.ball['pos'], 50, self.COLOR_GOAL)
            self.ball = self._create_ball(on_player=True)
            self._unlock_content()
            # sfx: score_sound()
        elif self.ball['pos'][0] > self.WIDTH - self.BALL_RADIUS and self.ball['pos'][1] > self.HEIGHT - goal_height: # Player Goal
            self.opponent_score += 1
            self.reward_this_step -= 2
            self._spawn_particles(self.ball['pos'], 50, self.COLOR_OPPONENT)
            self.ball = self._create_ball(on_player=True)
            # sfx: opponent_score_sound()

        # --- Player Catching Ball ---
        if not self.ball['on_player']:
            dist_to_player = np.linalg.norm(self.player_pos - self.ball['pos'])
            if dist_to_player < self.PLAYER_RADIUS + self.BALL_RADIUS:
                self.ball['on_player'] = True
                self.reward_this_step += 1
                # sfx: catch_sound()

        # --- Opponent Intercepting Ball ---
        if not self.ball['on_player']:
            for op in self.opponents:
                dist_to_op = np.linalg.norm(op['pos'] - self.ball['pos'])
                if dist_to_op < self.OPPONENT_RADIUS + self.BALL_RADIUS:
                    self.reward_this_step -= 1
                    self._spawn_particles(self.ball['pos'], 30, self.COLOR_OPPONENT)
                    self.ball = self._create_ball(on_player=False) # Reset to center
                    # sfx: intercept_sound()
                    break

        # --- Update Particles ---
        self._update_particles()
        if not self.ball['on_player']:
            self._spawn_particles(self.ball['pos'], 1, self.BALL_COLORS[self.ball['type']], trail=True)

    def _check_termination(self):
        if self.score >= self.WIN_SCORE or self.opponent_score >= self.LOSE_SCORE or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _unlock_content(self):
        if self.score >= 3 and 'gravity' not in GameEnv.unlocked_enchantments:
            GameEnv.unlocked_enchantments.append('gravity')
        if self.score >= 6 and 'speed' not in GameEnv.unlocked_enchantments:
            GameEnv.unlocked_enchantments.append('speed')
        # Visual change at score 9
        if self.score >= 9:
            self.COLOR_COURT = (255, 223, 0, 180) # Gold court for final point

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw stars
        for x, y, size in self.stars:
            pygame.draw.rect(self.screen, (200,200,255,100), (x, y, size, size))

        # Draw goals
        goal_height = self.HEIGHT / 2
        pygame.draw.rect(self.screen, self.COLOR_GOAL, (0, self.HEIGHT - goal_height, 5, goal_height))
        pygame.draw.rect(self.screen, self.COLOR_GOAL, (self.WIDTH - 5, self.HEIGHT - goal_height, 5, goal_height))
        
        # Draw court lines
        pygame.gfxdraw.line(self.screen, 0, self.HEIGHT - 1, self.WIDTH, self.HEIGHT - 1, self.COLOR_COURT)
        pygame.gfxdraw.line(self.screen, self.WIDTH // 2, self.HEIGHT, self.WIDTH // 2, self.HEIGHT - 20, self.COLOR_COURT)

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['radius']))

        # Draw opponents
        for op in self.opponents:
            pos_int = op['pos'].astype(int)
            self._draw_glow_circle(self.screen, pos_int, self.OPPONENT_RADIUS, self.COLOR_OPPONENT_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.OPPONENT_RADIUS, self.COLOR_OPPONENT)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.OPPONENT_RADIUS, self.COLOR_OPPONENT)

        # Draw player
        player_pos_int = self.player_pos.astype(int)
        self._draw_glow_circle(self.screen, player_pos_int, self.PLAYER_RADIUS, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

        # Draw trajectory line
        if self.ball['on_player']:
            start_pos = self.ball['pos']
            end_pos = start_pos + np.array([math.cos(self.aim_angle) * 60, math.sin(self.aim_angle) * 60])
            pygame.draw.line(self.screen, self.COLOR_TEXT, start_pos.astype(int), end_pos.astype(int), 2)

        # Draw ball
        ball_pos_int = self.ball['pos'].astype(int)
        ball_color = self.BALL_COLORS[self.ball['type']]
        self._draw_glow_circle(self.screen, ball_pos_int, self.BALL_RADIUS, (*ball_color, 60))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, ball_color)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, ball_color)

    def _render_ui(self):
        # Score display
        score_text = self.font_main.render(f"{self.score} - {self.opponent_score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 10))

        # Enchantment display
        enchant_text = self.unlocked_enchantments[self.current_enchantment_idx].upper()
        enchant_color = self.BALL_COLORS[self.unlocked_enchantments[self.current_enchantment_idx]]
        text_surf = self.font_small.render(enchant_text, True, enchant_color)
        
        icon_pos_x = self.player_pos[0]
        icon_pos_y = self.player_pos[1] - self.PLAYER_RADIUS - 30
        
        self.screen.blit(text_surf, (icon_pos_x - text_surf.get_width() / 2, icon_pos_y))

    def _spawn_particles(self, pos, count, color, trail=False):
        for _ in range(count):
            if trail:
                vel = (np.random.rand(2) - 0.5) * 1
                life = random.randint(10, 20)
                radius = random.uniform(1, 3)
            else:
                vel = (np.random.rand(2) - 0.5) * random.uniform(2, 8)
                life = random.randint(20, 40)
                radius = random.uniform(2, 5)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': life, 'radius': radius, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] *= 0.95
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0.5]

    def _draw_glow_circle(self, surf, pos, radius, color):
        glow_radius = int(radius * 1.8)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, color, (glow_radius, glow_radius), glow_radius)
        surf.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _get_info(self):
        return {
            "score": self.score,
            "opponent_score": self.opponent_score,
            "steps": self.steps,
            "ball_type": self.ball['type']
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # For a display window, comment out the os.environ line at the top
    # and instantiate GameEnv with render_mode="human"
    if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
        # In headless mode, we can't create a display, so we can't run the interactive test
        print("Running in headless mode. Interactive test skipped.")
        # Basic validation can still run
        env = GameEnv()
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray) and obs.shape == env.observation_space.shape
        print("Headless step test passed.")
        env.close()
    else:
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Gravity-Flipping Netball")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        while running:
            movement = 0 # No-op
            space_held = 0
            shift_held = 0

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            if keys[pygame.K_SPACE]:
                space_held = 1
                
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift_held = 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
                    total_reward = 0

            action = [movement, space_held, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                print(f"Episode finished. Total reward: {total_reward}. Info: {info}")
                running = False # End after one episode for automated testing, or comment out to play again

            # Convert observation back to a surface for display
            # The observation is (H, W, C), but pygame wants (W, H) for surface creation
            # and the array is transposed from pygame's internal format. So we need to transpose back.
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(env.FPS)
            
        env.close()