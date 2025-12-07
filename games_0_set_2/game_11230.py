import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:49:05.190250
# Source Brief: brief_01230.md
# Brief Index: 1230
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where two clockwork gladiators fight in a steampunk arena.
    The player controls one gladiator and must defeat the AI opponent using standard
    attacks and a time-stopping ability. The environment is designed for visual
    quality and satisfying gameplay feel.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Engage in a steampunk duel as a clockwork gladiator. Defeat your opponent using precise attacks and a powerful time-stopping ability."
    user_guide = "Controls: Use ↑↓←→ arrow keys to move. Press space to attack and shift to activate your time-stopping ability."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1500

    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_BG_GEAR = (40, 45, 50)
    COLOR_BG_PIPE = (50, 55, 60)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255, 50)
    COLOR_OPPONENT = (255, 50, 50)
    COLOR_OPPONENT_GLOW = (255, 50, 50, 50)
    COLOR_HEALTH_BG = (80, 20, 20)
    COLOR_HEALTH_FG = (50, 200, 50)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_GOLD = (255, 215, 0)
    COLOR_COOLDOWN_OVERLAY = (0, 0, 0, 150)
    COLOR_TIME_STOP_EFFECT = (255, 215, 0, 40)

    # Player settings
    PLAYER_SPEED = 5
    PLAYER_HEALTH = 100
    PLAYER_RADIUS = 15
    ATTACK_RANGE = 45
    ATTACK_ARC = math.pi / 2 # 90 degrees
    ATTACK_DAMAGE = 10
    ATTACK_COOLDOWN = 15 # frames
    TIME_STOP_DURATION = 90 # frames
    TIME_STOP_COOLDOWN = 450 # frames

    # Opponent AI settings
    OPPONENT_SPEED = 3
    OPPONENT_HEALTH = 100
    OPPONENT_RADIUS = 15
    OPPONENT_ATTACK_COOLDOWN = 30
    OPPONENT_ATTACK_RANGE = 45
    OPPONENT_ATTACK_DAMAGE = 8
    AI_STATE_DURATION_MIN = 30
    AI_STATE_DURATION_MAX = 90

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_health = None
        self.player_attack_cooldown = 0
        self.player_time_stop_cooldown = 0
        self.player_facing_angle = 0
        self.player_attacking = 0
        
        self.opponent_pos = None
        self.opponent_health = None
        self.opponent_attack_cooldown = 0
        self.opponent_facing_angle = 0
        self.opponent_attacking = 0

        self.ai_state = "PATROL"
        self.ai_state_timer = 0
        self.ai_patrol_target = None
        self.ai_attack_duration_bonus = 0

        self.time_stop_timer = 0
        self.particles = []
        self.events = [] # For reward calculation

        # Track button presses
        self.prev_space_held = False
        self.prev_shift_held = False
        
        # Pre-render background for performance
        self._background_surface = self._create_background()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = np.array([100.0, self.SCREEN_HEIGHT / 2.0])
        self.player_health = self.PLAYER_HEALTH
        self.player_attack_cooldown = 0
        self.player_time_stop_cooldown = 0
        self.player_facing_angle = 0
        self.player_attacking = 0

        self.opponent_pos = np.array([self.SCREEN_WIDTH - 100.0, self.SCREEN_HEIGHT / 2.0])
        self.opponent_health = self.OPPONENT_HEALTH
        self.opponent_attack_cooldown = 0
        self.opponent_facing_angle = math.pi
        self.opponent_attacking = 0

        self.ai_state = "PATROL"
        self.ai_state_timer = self.AI_STATE_DURATION_MAX
        self._set_new_patrol_target()
        self.ai_attack_duration_bonus = 0

        self.time_stop_timer = 0
        self.particles = []
        self.events = []

        self.prev_space_held = True # Prevent action on first frame
        self.prev_shift_held = True

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.events.clear()
        self.steps += 1
        
        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        # --- Update Game State ---
        self._update_cooldowns()
        self._update_player(movement, space_press, shift_press)
        if self.time_stop_timer <= 0:
            self._update_opponent()
        
        self._update_particles()
        
        # --- Difficulty Scaling ---
        if self.steps % 500 == 0 and self.ai_attack_duration_bonus < 30:
            self.ai_attack_duration_bonus += 1

        # --- Calculate Reward and Termination ---
        reward = self._calculate_reward()
        self.score += reward
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _update_cooldowns(self):
        is_time_stopped = self.time_stop_timer > 0
        
        if self.player_attack_cooldown > 0: self.player_attack_cooldown -= 1
        if self.player_time_stop_cooldown > 0: self.player_time_stop_cooldown -= 1
        if self.player_attacking > 0: self.player_attacking -=1

        if not is_time_stopped:
            if self.opponent_attack_cooldown > 0: self.opponent_attack_cooldown -= 1
            if self.opponent_attacking > 0: self.opponent_attacking -=1

        if self.time_stop_timer > 0: self.time_stop_timer -= 1

    def _update_player(self, movement, space_press, shift_press):
        # Movement
        move_vec = np.array([0.0, 0.0])
        if movement == 1: move_vec[1] = -1
        elif movement == 2: move_vec[1] = 1
        elif movement == 3: move_vec[0] = -1
        elif movement == 4: move_vec[0] = 1
        
        if np.linalg.norm(move_vec) > 0:
            self.player_facing_angle = math.atan2(move_vec[1], move_vec[0])
            self.player_pos += move_vec * self.PLAYER_SPEED

        # Clamp position
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)

        # Actions
        if space_press and self.player_attack_cooldown <= 0:
            self._player_attack()
        
        if shift_press and self.player_time_stop_cooldown <= 0:
            self._player_time_stop()
            
    def _player_attack(self):
        self.player_attack_cooldown = self.ATTACK_COOLDOWN
        self.player_attacking = 5 # Attack animation lasts 5 frames
        # sfx: player_sword_swing.wav
        
        dist = np.linalg.norm(self.opponent_pos - self.player_pos)
        if dist < self.ATTACK_RANGE:
            angle_to_opponent = math.atan2(self.opponent_pos[1] - self.player_pos[1], self.opponent_pos[0] - self.player_pos[0])
            angle_diff = abs((self.player_facing_angle - angle_to_opponent + math.pi) % (2 * math.pi) - math.pi)
            
            if angle_diff <= self.ATTACK_ARC / 2:
                self.opponent_health -= self.ATTACK_DAMAGE
                self.events.append("player_hit")
                self._create_sparks(self.opponent_pos, self.COLOR_GOLD)
                # sfx: hit_connect.wav

    def _player_time_stop(self):
        self.player_time_stop_cooldown = self.TIME_STOP_COOLDOWN
        self.time_stop_timer = self.TIME_STOP_DURATION
        # sfx: time_stop_activate.wav

    def _update_opponent(self):
        self.ai_state_timer -= 1
        
        dist_to_player = np.linalg.norm(self.player_pos - self.opponent_pos)
        
        # State transitions
        if self.ai_state == "PATROL" and dist_to_player < self.OPPONENT_ATTACK_RANGE * 3:
            self.ai_state = "ATTACK"
            self.ai_state_timer = self.AI_STATE_DURATION_MAX + self.ai_attack_duration_bonus
        elif self.ai_state == "ATTACK" and dist_to_player > self.OPPONENT_ATTACK_RANGE * 4:
            self.ai_state = "PATROL"
            self.ai_state_timer = self.AI_STATE_DURATION_MAX
        elif self.ai_state_timer <= 0:
            self.ai_state = "PATROL" if self.ai_state == "ATTACK" else "ATTACK"
            self.ai_state_timer = random.randint(self.AI_STATE_DURATION_MIN, self.AI_STATE_DURATION_MAX)
            if self.ai_state == "ATTACK": self.ai_state_timer += self.ai_attack_duration_bonus

        # State actions
        if self.ai_state == "PATROL":
            if np.linalg.norm(self.ai_patrol_target - self.opponent_pos) < 20:
                self._set_new_patrol_target()
            
            move_vec = self.ai_patrol_target - self.opponent_pos
            norm = np.linalg.norm(move_vec)
            if norm > 0:
                self.opponent_pos += (move_vec / norm) * self.OPPONENT_SPEED
                self.opponent_facing_angle = math.atan2(move_vec[1], move_vec[0])

        elif self.ai_state == "ATTACK":
            move_vec = self.player_pos - self.opponent_pos
            norm = np.linalg.norm(move_vec)
            if norm > self.OPPONENT_ATTACK_RANGE * 0.8: # Chase player
                self.opponent_pos += (move_vec / norm) * self.OPPONENT_SPEED
                self.opponent_facing_angle = math.atan2(move_vec[1], move_vec[0])
            
            if dist_to_player < self.OPPONENT_ATTACK_RANGE and self.opponent_attack_cooldown <= 0:
                self._opponent_attack()

        # Clamp position
        self.opponent_pos[0] = np.clip(self.opponent_pos[0], self.OPPONENT_RADIUS, self.SCREEN_WIDTH - self.OPPONENT_RADIUS)
        self.opponent_pos[1] = np.clip(self.opponent_pos[1], self.OPPONENT_RADIUS, self.SCREEN_HEIGHT - self.OPPONENT_RADIUS)
    
    def _opponent_attack(self):
        self.opponent_attack_cooldown = self.OPPONENT_ATTACK_COOLDOWN
        self.opponent_attacking = 5
        # sfx: opponent_sword_swing.wav
        self.player_health -= self.OPPONENT_ATTACK_DAMAGE
        self.events.append("opponent_hit")
        self._create_sparks(self.player_pos, self.COLOR_OPPONENT)
        # sfx: player_damage.wav

    def _set_new_patrol_target(self):
        self.ai_patrol_target = np.array([
            random.uniform(self.OPPONENT_RADIUS, self.SCREEN_WIDTH - self.OPPONENT_RADIUS),
            random.uniform(self.OPPONENT_RADIUS, self.SCREEN_HEIGHT - self.OPPONENT_RADIUS)
        ])

    def _calculate_reward(self):
        reward = 0
        if "player_hit" in self.events:
            reward += 1.0  # Event-based reward for landing a hit
            reward += 0.1  # Continuous feedback for dealing damage
        if "opponent_hit" in self.events:
            reward -= 1.0  # Event-based penalty
            reward -= 0.1  # Continuous penalty for taking damage
        
        if self.player_health <= 0:
            reward -= 100
        if self.opponent_health <= 0:
            reward += 100
        
        return reward

    def _check_termination(self):
        if self.player_health <= 0 or self.opponent_health <= 0:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        # Draw background
        self.screen.blit(self._background_surface, (0, 0))
        
        # Render game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "opponent_health": self.opponent_health,
            "time_stop_cooldown": self.player_time_stop_cooldown,
            "time_stop_active": self.time_stop_timer > 0,
        }

    def _render_game(self):
        # Draw gladiators
        self._render_gladiator(self.player_pos, self.PLAYER_RADIUS, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW, self.player_facing_angle, self.player_attacking > 0)
        self._render_gladiator(self.opponent_pos, self.OPPONENT_RADIUS, self.COLOR_OPPONENT, self.COLOR_OPPONENT_GLOW, self.opponent_facing_angle, self.opponent_attacking > 0)
        
        # Draw particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            pygame.draw.circle(self.screen, p['color'], p['pos'], p['size'])

        # Draw time stop effect
        if self.time_stop_timer > 0:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_TIME_STOP_EFFECT)
            self.screen.blit(overlay, (0,0))
    
    def _render_gladiator(self, pos, radius, color, glow_color, angle, is_attacking):
        x, y = int(pos[0]), int(pos[1])
        
        # Glow effect
        glow_radius = int(radius * 2.5)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, glow_color, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (x - glow_radius, y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Body
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
        
        # Central gear
        gear_radius = int(radius * 0.6)
        num_teeth = 8
        gear_angle = (self.steps * 0.1) % (2 * math.pi)
        for i in range(num_teeth):
            a = gear_angle + i * (2 * math.pi / num_teeth)
            p1 = (x + math.cos(a) * gear_radius, y + math.sin(a) * gear_radius)
            p2 = (x + math.cos(a) * (gear_radius + 3), y + math.sin(a) * (gear_radius + 3))
            pygame.draw.line(self.screen, self.COLOR_BG, p1, p2, 2)

        # Sword
        sword_length = radius * 1.8
        sword_offset = radius * 0.5
        
        # Lunge animation
        lunge = 0
        if is_attacking:
            lunge = 15 * math.sin((5 - self.player_attacking if color == self.COLOR_PLAYER else 5 - self.opponent_attacking) / 5 * math.pi)

        start_x = x + math.cos(angle) * sword_offset
        start_y = y + math.sin(angle) * sword_offset
        end_x = x + math.cos(angle) * (sword_length + lunge)
        end_y = y + math.sin(angle) * (sword_length + lunge)
        pygame.draw.line(self.screen, self.COLOR_UI_TEXT, (int(start_x), int(start_y)), (int(end_x), int(end_y)), 4)

    def _render_ui(self):
        # Health bars
        self._draw_health_bar(self.player_pos, self.player_health, self.PLAYER_HEALTH, self.COLOR_PLAYER)
        self._draw_health_bar(self.opponent_pos, self.opponent_health, self.OPPONENT_HEALTH, self.COLOR_OPPONENT)
        
        # Score and Steps
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))

        # Cooldown icons
        self._draw_cooldown_icon(self.SCREEN_WIDTH / 2 - 40, self.SCREEN_HEIGHT - 45, self.ATTACK_COOLDOWN, self.player_attack_cooldown, "ATK")
        self._draw_cooldown_icon(self.SCREEN_WIDTH / 2 + 10, self.SCREEN_HEIGHT - 45, self.TIME_STOP_COOLDOWN, self.player_time_stop_cooldown, "TIME")
        
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            msg = "YOU WIN!" if self.opponent_health <= 0 else "YOU LOSE"
            if self.player_health <= 0 and self.opponent_health <= 0: msg = "DRAW"
            elif self.steps >= self.MAX_STEPS and self.player_health > 0 and self.opponent_health > 0: msg = "TIME UP"
            
            end_text = self.font_large.render(msg, True, self.COLOR_GOLD)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _draw_health_bar(self, pos, current_hp, max_hp, color):
        bar_width = 40
        bar_height = 5
        x = int(pos[0] - bar_width / 2)
        y = int(pos[1] - self.PLAYER_RADIUS - 15)
        
        health_ratio = max(0, current_hp / max_hp)
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (x, y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (x, y, int(bar_width * health_ratio), bar_height))
        pygame.draw.rect(self.screen, color, (x, y, bar_width, bar_height), 1)

    def _draw_cooldown_icon(self, x, y, max_cd, current_cd, text):
        size = 35
        rect = pygame.Rect(x, y, size, size)
        
        pygame.draw.rect(self.screen, self.COLOR_BG_PIPE, rect)
        pygame.draw.rect(self.screen, self.COLOR_GOLD, rect, 2)
        
        label = self.font_small.render(text, True, self.COLOR_UI_TEXT)
        text_rect = label.get_rect(center=rect.center)
        self.screen.blit(label, text_rect)
        
        if current_cd > 0:
            cd_ratio = current_cd / max_cd
            overlay_height = int(size * cd_ratio)
            overlay_surf = pygame.Surface((size, overlay_height), pygame.SRCALPHA)
            overlay_surf.fill(self.COLOR_COOLDOWN_OVERLAY)
            self.screen.blit(overlay_surf, (x, y + size - overlay_height))

    def _create_background(self):
        bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        bg.fill(self.COLOR_BG)
        
        # Draw some large, static gears and pipes
        for _ in range(10):
            r = random.randint(50, 200)
            x = random.randint(-r, self.SCREEN_WIDTH + r)
            y = random.randint(-r, self.SCREEN_HEIGHT + r)
            pygame.gfxdraw.aacircle(bg, x, y, r, self.COLOR_BG_GEAR)
        for _ in range(5):
            x1, y1 = random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT)
            x2, y2 = random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT)
            pygame.draw.line(bg, self.COLOR_BG_PIPE, (x1, y1), (x2, y2), random.randint(10, 30))
        return bg

    def _create_sparks(self, pos, color):
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': pos.copy(),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'life': random.randint(10, 20),
                'color': color,
                'size': random.randint(1, 3)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to test and play the game.
    # It requires a graphical display.
    os.environ["SDL_VIDEODRIVER"] = "" # Use default video driver
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Clockwork Gladiator Arena")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The observation is already the rendered screen, so we just need to display it.
        # Pygame uses (width, height), but our obs is (height, width, 3).
        # We need to transpose it back for pygame.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}")
            print("Resetting environment...")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause before restarting
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(GameEnv.FPS)
        
    env.close()