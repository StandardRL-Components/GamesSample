import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:40:44.877366
# Source Brief: brief_00562.md
# Brief Index: 562
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    def __init__(self, x, y, color, min_vel=-3, max_vel=3, gravity=0.1, life=20):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(random.uniform(min_vel, max_vel), random.uniform(min_vel, max_vel))
        self.color = color
        self.gravity = gravity
        self.life = life
        self.max_life = life

    def update(self):
        self.pos += self.vel
        self.vel.y += self.gravity
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            radius = int(5 * (self.life / self.max_life))
            if radius > 0:
                # Create a temporary surface for alpha blending
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*self.color, alpha), (radius, radius), radius)
                surface.blit(temp_surf, (int(self.pos.x - radius), int(self.pos.y - radius)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "A turn-based physics shooter. Aim your meson, match the target's color, and destroy your opponent's targets before they destroy yours."
    )
    user_guide = (
        "Controls: ↑↓←→ to aim, hold Space to charge power and release to fire. Press Shift to cycle projectile color."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30

        # Colors
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_FIELD = (25, 30, 50)
        self.COLOR_DIVIDER = (50, 60, 100)
        self.COLOR_PLAYER_AREA = (30, 35, 60)
        self.COLOR_OPPONENT_AREA = (35, 30, 60)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_ACCENT = (100, 120, 255)
        self.MESON_COLORS = [(255, 80, 80), (80, 255, 80), (80, 150, 255)] # Red, Green, Blue
        self.TARGET_COLORS = self.MESON_COLORS

        # Game constants
        self.GRAVITY = 0.2
        self.MAX_CHARGE = 100
        self.MIN_POWER = 5
        self.MAX_POWER = 18
        self.MAX_STEPS = 2500
        self.WIN_SCORE = 10
        self.TURN_TIME_LIMIT = 120 # 4 seconds at 30fps

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 24)
        self.font_title = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 50)
        
        # Persistent state
        self.league = 1
        
        # Initialize state variables to be reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_score = 0
        self.opponent_score = 0
        self.game_phase = "PLAYER_AIM"
        self.player_aim_angle = 0
        self.player_charge = 0
        self.player_meson_color_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.turn_timer = 0
        self.opponent_ai_timer = 0
        self.opponent_meson_speed = 2.0
        self.mesons = []
        self.particles = []
        self.targets_player = []
        self.targets_opponent = []
        self.reward_buffer = 0.0
        self.win_status = ""

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.player_score = 0
        self.opponent_score = 0
        self.score = 0 # Overall episode score, for info
        self.game_over = False
        self.win_status = ""
        
        self.game_phase = "PLAYER_AIM"
        self.player_aim_angle = -math.pi / 4
        self.player_charge = 0
        self.player_meson_color_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self.opponent_meson_speed = 2.0 + (self.league - 1) * 0.2
        self.opponent_ai_timer = self.np_random.integers(30, 60)

        self.mesons = []
        self.particles = []
        self.targets_player = self._generate_targets(is_player_side=True)
        self.targets_opponent = self._generate_targets(is_player_side=False)
        
        self.turn_timer = self.TURN_TIME_LIMIT
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_buffer = 0.0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        if self.game_phase == "PLAYER_AIM":
            self._handle_player_turn(action)
        elif self.game_phase == "OPPONENT_AIM":
            self._handle_opponent_turn()

        self._update_mesons()
        self._update_particles()
        
        self.steps += 1
        
        terminated, terminal_reward = self._check_and_apply_terminal_conditions()
        self.reward_buffer += terminal_reward
        
        reward = self.reward_buffer
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_turn(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self.turn_timer -= 1

        # 1. Handle aiming
        if movement == 1: self.player_aim_angle -= 0.05 # Up
        elif movement == 2: self.player_aim_angle += 0.05 # Down
        elif movement == 3: self.player_aim_angle -= 0.01 # Left (fine tune)
        elif movement == 4: self.player_aim_angle += 0.01 # Right (fine tune)
        self.player_aim_angle = np.clip(self.player_aim_angle, -math.pi * 0.9, -math.pi * 0.1)

        # 2. Handle color cycling
        if shift_held and not self.prev_shift_held:
            self.player_meson_color_idx = (self.player_meson_color_idx + 1) % len(self.MESON_COLORS)
            # sfx: color_cycle.wav

        # 3. Handle charging
        if space_held:
            self.player_charge = min(self.MAX_CHARGE, self.player_charge + 5)
            # sfx: charge_loop.wav (looping)

        # 4. Handle launch
        if not space_held and self.prev_space_held and self.player_charge > 0:
            power_ratio = self.player_charge / self.MAX_CHARGE
            power = self.MIN_POWER + (self.MAX_POWER - self.MIN_POWER) * power_ratio
            self._launch_meson(is_player=True, power=power, angle=self.player_aim_angle, color_idx=self.player_meson_color_idx)
            self.player_charge = 0
            self.game_phase = "IN_PLAY"
            # sfx: launch.wav

        # 5. Handle turn timeout
        if self.turn_timer <= 0:
            self.reward_buffer -= 0.1 # Miss penalty
            self._switch_to_opponent_turn()
            # sfx: timeout_error.wav

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _handle_opponent_turn(self):
        self.opponent_ai_timer -= 1
        if self.opponent_ai_timer <= 0:
            living_targets = [t for t in self.targets_player if t['alive']]
            if not living_targets:
                 self._switch_to_player_turn() # No targets to shoot at
                 return

            target = self.np_random.choice(living_targets)
            target_pos = pygame.Vector2(target['rect'].center)
            
            # Simple AI: aim directly at a target with random power
            start_pos = pygame.Vector2(self.WIDTH - 40, self.HEIGHT - 40)
            angle = math.atan2(target_pos.y - start_pos.y, target_pos.x - start_pos.x)
            angle = np.clip(angle, -math.pi * 0.9, -math.pi * 0.1)
            
            power = self.np_random.uniform(self.MIN_POWER, self.MAX_POWER) * 0.8 # Opponent is slightly weaker
            color_idx = self.np_random.integers(len(self.MESON_COLORS))
            
            self._launch_meson(is_player=False, power=power, angle=angle, color_idx=color_idx)
            self.game_phase = "IN_PLAY"
            # sfx: opponent_launch.wav
    
    def _launch_meson(self, is_player, power, angle, color_idx):
        if is_player:
            pos = pygame.Vector2(40, self.HEIGHT - 40)
            vel = pygame.Vector2(power * math.cos(angle), power * math.sin(angle))
        else:
            pos = pygame.Vector2(self.WIDTH - 40, self.HEIGHT - 40)
            vel = pygame.Vector2(power * math.cos(angle), power * math.sin(angle))

        self.mesons.append({
            'pos': pos, 'vel': vel, 'color_idx': color_idx, 'owner': 'player' if is_player else 'opponent', 'trail': []
        })

    def _update_mesons(self):
        for meson in self.mesons[:]:
            meson['trail'].append(meson['pos'].copy())
            if len(meson['trail']) > 15:
                meson['trail'].pop(0)

            meson['pos'] += meson['vel']
            meson['vel'].y += self.GRAVITY

            # Check for out of bounds / miss
            if not (0 < meson['pos'].x < self.WIDTH and meson['pos'].y < self.HEIGHT):
                self.reward_buffer -= 0.1
                self.mesons.remove(meson)
                if meson['owner'] == 'player': self._switch_to_opponent_turn()
                else: self._switch_to_player_turn()
                # sfx: miss_whoosh.wav
                continue

            # Check for target collision
            targets_to_check = self.targets_opponent if meson['owner'] == 'player' else self.targets_player
            hit = False
            for target in targets_to_check:
                if target['alive'] and target['rect'].collidepoint(meson['pos']):
                    target['alive'] = False
                    hit = True
                    self.reward_buffer += 0.1 # Base hit reward
                    self.reward_buffer += 1.0  # Target destruction reward
                    
                    if meson['owner'] == 'player': self.player_score += 1
                    else: self.opponent_score += 1
                    
                    # Color match bonus
                    if meson['color_idx'] == target['color_idx']:
                        self.reward_buffer += 5.0
                        if meson['owner'] == 'player': self.player_score += 2 # Bonus points
                        else: self.opponent_score += 2
                        self._create_explosion(meson['pos'], self.MESON_COLORS[meson['color_idx']], 50, 5)
                        # sfx: explosion_large.wav
                    else:
                        self._create_explosion(meson['pos'], self.MESON_COLORS[meson['color_idx']], 20, 3)
                        # sfx: explosion_small.wav

                    break # only one hit per frame

            if hit:
                self.mesons.remove(meson)
                if meson['owner'] == 'player': self._switch_to_opponent_turn()
                else: self._switch_to_player_turn()

    def _create_explosion(self, pos, color, count, max_vel):
        for _ in range(count):
            self.particles.append(Particle(pos.x, pos.y, color, -max_vel, max_vel))
    
    def _update_particles(self):
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

    def _switch_to_player_turn(self):
        self.game_phase = "PLAYER_AIM"
        self.turn_timer = self.TURN_TIME_LIMIT
        # Check if opponent side is clear
        if not any(t['alive'] for t in self.targets_opponent):
            self.targets_opponent = self._generate_targets(is_player_side=False)

    def _switch_to_opponent_turn(self):
        self.game_phase = "OPPONENT_AIM"
        self.opponent_ai_timer = self.np_random.integers(30, 60)
        # Check if player side is clear
        if not any(t['alive'] for t in self.targets_player):
            self.targets_player = self._generate_targets(is_player_side=True)

    def _check_and_apply_terminal_conditions(self):
        if self.game_over: return True, 0.0

        if self.player_score >= self.WIN_SCORE:
            self.game_over = True
            self.win_status = "YOU WON"
            self.league += 1
            return True, 100.0
        
        if self.opponent_score >= self.WIN_SCORE:
            self.game_over = True
            self.win_status = "YOU LOST"
            return True, -100.0

        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            if self.player_score > self.opponent_score:
                self.win_status = "YOU WON (TIME)"
                self.league += 1
                return True, 100.0
            elif self.opponent_score > self.player_score:
                self.win_status = "YOU LOST (TIME)"
                return True, -100.0
            else:
                self.win_status = "DRAW"
                return True, 0.0
        
        return False, 0.0

    def _generate_targets(self, is_player_side):
        targets = []
        num_targets = min(8, 2 + self.league)
        
        for i in range(num_targets):
            size = self.np_random.integers(15, max(16, 40 - self.league * 2))
            
            if is_player_side:
                x = self.np_random.integers(20, self.WIDTH // 2 - 40)
            else:
                x = self.np_random.integers(self.WIDTH // 2 + 40, self.WIDTH - 20)
            
            y = self.np_random.integers(50, self.HEIGHT - 80)
            
            color_idx = self.np_random.integers(len(self.TARGET_COLORS))
            
            targets.append({
                'rect': pygame.Rect(x - size // 2, y - size // 2, size, size),
                'color_idx': color_idx,
                'alive': True
            })
        return targets

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Field
        field_rect = pygame.Rect(10, 10, self.WIDTH - 20, self.HEIGHT - 20)
        pygame.draw.rect(self.screen, self.COLOR_FIELD, field_rect, border_radius=10)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_AREA, (10, 10, self.WIDTH // 2 - 10, self.HEIGHT - 20), border_top_left_radius=10, border_bottom_left_radius=10)
        pygame.draw.rect(self.screen, self.COLOR_OPPONENT_AREA, (self.WIDTH // 2, 10, self.WIDTH // 2 - 10, self.HEIGHT - 20), border_top_right_radius=10, border_bottom_right_radius=10)
        pygame.draw.line(self.screen, self.COLOR_DIVIDER, (self.WIDTH // 2, 10), (self.WIDTH // 2, self.HEIGHT - 10), 3)

        # Targets
        for target in self.targets_player + self.targets_opponent:
            if target['alive']:
                color = self.TARGET_COLORS[target['color_idx']]
                pygame.draw.rect(self.screen, color, target['rect'], border_radius=4)
                pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), target['rect'], width=2, border_radius=4)
        
        # Aiming preview
        if self.game_phase == "PLAYER_AIM" and self.player_charge > 0:
            power_ratio = self.player_charge / self.MAX_CHARGE
            power = self.MIN_POWER + (self.MAX_POWER - self.MIN_POWER) * power_ratio
            pos = pygame.Vector2(40, self.HEIGHT - 40)
            vel = pygame.Vector2(power * math.cos(self.player_aim_angle), power * math.sin(self.player_aim_angle))
            for _ in range(30):
                pos += vel
                vel.y += self.GRAVITY
                if _ % 3 == 0:
                    pygame.draw.circle(self.screen, self.COLOR_UI_ACCENT, (int(pos.x), int(pos.y)), 1)
        
        # Mesons and trails
        for meson in self.mesons:
            color = self.MESON_COLORS[meson['color_idx']]
            # Trail
            for i, pos in enumerate(meson['trail']):
                alpha = int(255 * (i / len(meson['trail'])))
                radius = int(3 * (i / len(meson['trail'])))
                if radius > 0:
                    temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, (*color, alpha), (radius, radius), radius)
                    self.screen.blit(temp_surf, (int(pos.x - radius), int(pos.y - radius)))
            # Glow
            glow_radius = 12
            temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*color, 60), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(temp_surf, (int(meson['pos'].x - glow_radius), int(meson['pos'].y - glow_radius)))
            # Core
            pygame.gfxdraw.aacircle(self.screen, int(meson['pos'].x), int(meson['pos'].y), 5, color)
            pygame.gfxdraw.filled_circle(self.screen, int(meson['pos'].x), int(meson['pos'].y), 5, color)

        # Particles
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Scores
        score_text = f"YOU {self.player_score} - {self.opponent_score} OPP"
        score_surf = self.font_title.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (self.WIDTH // 2 - score_surf.get_width() // 2, 15))

        # League
        league_text = f"LEAGUE: {self.league}"
        league_surf = self.font_main.render(league_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(league_surf, (20, 15))

        # Player turn UI
        if self.game_phase == "PLAYER_AIM":
            # Color indicator
            color = self.MESON_COLORS[self.player_meson_color_idx]
            pygame.draw.rect(self.screen, color, (20, self.HEIGHT - 50, 40, 10), border_radius=3)
            color_text = self.font_main.render("COLOR", True, self.COLOR_UI_TEXT)
            self.screen.blit(color_text, (20, self.HEIGHT - 70))

            # Charge bar
            charge_ratio = self.player_charge / self.MAX_CHARGE
            bar_width = 100
            pygame.draw.rect(self.screen, self.COLOR_DIVIDER, (80, self.HEIGHT - 50, bar_width, 10), border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_UI_ACCENT, (80, self.HEIGHT - 50, int(bar_width * charge_ratio), 10), border_radius=3)
            charge_text = self.font_main.render("POWER", True, self.COLOR_UI_TEXT)
            self.screen.blit(charge_text, (80, self.HEIGHT - 70))

            # Turn timer
            timer_ratio = self.turn_timer / self.TURN_TIME_LIMIT
            timer_color = (255, 255, 100) if timer_ratio > 0.5 else (255, 100, 100)
            pygame.draw.rect(self.screen, timer_color, (20, self.HEIGHT - 30, int(160 * timer_ratio), 5), border_radius=2)
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            status_surf = self.font_game_over.render(self.win_status, True, self.COLOR_UI_TEXT)
            self.screen.blit(status_surf, (self.WIDTH // 2 - status_surf.get_width() // 2, self.HEIGHT // 2 - 40))
            
            score_summary = f"Final Score: {self.player_score} - {self.opponent_score}"
            summary_surf = self.font_main.render(score_summary, True, self.COLOR_UI_TEXT)
            self.screen.blit(summary_surf, (self.WIDTH // 2 - summary_surf.get_width() // 2, self.HEIGHT // 2 + 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_score": self.player_score,
            "opponent_score": self.opponent_score,
            "league": self.league,
            "game_phase": self.game_phase,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # Use a display for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    pygame.display.init()
    pygame.display.set_caption("Meson Sport")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    movement_action = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
    space_action = 0    # 0=released, 1=held
    shift_action = 0    # 0=released, 1=held

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            
            # Keyboard handling for manual play
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: space_action = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_action = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE: space_action = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_action = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        else: movement_action = 0
        
        # The action space is MultiDiscrete, so we build the action array
        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Player Score: {info['player_score']}, Opp Score: {info['opponent_score']}")

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(env.FPS)
        
        # After a game ends, reset to play again
        if done:
            print(f"Game Over. Status: {env.win_status}. Final Score: {info['player_score']}-{info['opponent_score']}. New League: {env.league}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            done = False
            
    env.close()