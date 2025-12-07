import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:21:35.702887
# Source Brief: brief_01017.md
# Brief Index: 1017
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "A futuristic boxing game where you must manage your momentum to defeat a series of opponents. "
        "Land hits to build momentum, but be careful not to let it run out."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys for jabs and ↑↓ for hooks. "
        "Not pressing any key will result in a block."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30

        # Game constants
        self.MAX_STEPS = 1000
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_MAX_MOMENTUM = 100
        self.OPPONENT_HEALTHS = [100, 110, 120]
        self.MOMENTUM_DECAY_RATE = 1.0 / self.FPS # 1% per second
        self.PLAYER_POS = (self.WIDTH // 4, self.HEIGHT // 2 + 50)
        self.OPPONENT_POS = (self.WIDTH * 3 // 4, self.HEIGHT // 2 + 50)

        # Attack properties: {damage, momentum_gain, cooldown_frames}
        self.ATTACKS = {
            "jab": {"damage": 5, "momentum": 8, "cooldown": 20},
            "hook": {"damage": 15, "momentum": 15, "cooldown": 45},
        }

        # Colors
        self.COLOR_BG = (10, 5, 30)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_OPPONENT = (255, 60, 60)
        self.COLOR_MOMENTUM = (0, 128, 255)
        self.COLOR_ATTACK = (255, 255, 0)
        self.COLOR_UI_BG = (40, 40, 40)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_RING = (100, 100, 120)

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont('Consolas', 16, bold=True)
        self.font_large = pygame.font.SysFont('Consolas', 48, bold=True)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_momentum = 0
        self.is_momentum_maxed = False
        self.current_opponent_idx = 0
        self.current_opponent_health = 0
        self.player_action = "idle"
        self.player_action_timer = 0
        self.opponent_action = "idle"
        self.opponent_action_timer = 0
        self.opponent_pattern_index = 0
        self.opponent_attack_pattern = ['jab', 'hook', 'block']
        self.particles = []
        self.hit_flashes = []
        
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_momentum = 50.0
        self.is_momentum_maxed = False
        
        self.current_opponent_idx = 0
        self.current_opponent_health = float(self.OPPONENT_HEALTHS[self.current_opponent_idx])
        
        self.player_action = "idle"
        self.player_action_timer = 0
        
        self.opponent_action = "idle"
        self.opponent_action_timer = 60 # Start with a delay
        self.opponent_pattern_index = 0
        
        self.particles = []
        self.hit_flashes = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.steps += 1
        reward = 0.0

        # -- Update Timers and Effects --
        self.player_action_timer = max(0, self.player_action_timer - 1)
        self.opponent_action_timer = max(0, self.opponent_action_timer - 1)
        self._update_effects()

        # Reset action states if timers run out
        if self.player_action_timer == 0: self.player_action = "idle"
        if self.opponent_action_timer == 0: self.opponent_action = "idle"

        # -- Handle Player Action --
        hit_landed = False
        attack_blocked = False

        if self.player_action_timer == 0:
            player_attack_type = None
            if movement == 1 or movement == 2: player_attack_type = "hook"
            elif movement == 3 or movement == 4: player_attack_type = "jab"
            
            if player_attack_type:
                # // Sound effect: "SWOOSH"
                attack_info = self.ATTACKS[player_attack_type]
                self.player_action = player_attack_type
                self.player_action_timer = attack_info["cooldown"]

                if self.opponent_action == "block":
                    attack_blocked = True
                    self._create_hit_flash(self.OPPONENT_POS, self.COLOR_ATTACK, 15)
                    # // Sound effect: "BLOCK"
                else:
                    hit_landed = True
                    self.current_opponent_health -= attack_info["damage"]
                    self.player_momentum += attack_info["momentum"]
                    self._create_particles(self.OPPONENT_POS, self.COLOR_OPPONENT)
                    self._create_hit_flash(self.OPPONENT_POS, self.COLOR_OPPONENT, 40)
                    # // Sound effect: "HIT_SUCCESS"
            else: # movement == 0 is block
                self.player_action = "block"
                self.player_action_timer = 5 # Brief block visual

        # -- Handle Opponent AI --
        if self.opponent_action_timer == 0:
            action_name = self.opponent_attack_pattern[self.opponent_pattern_index]
            self.opponent_pattern_index = (self.opponent_pattern_index + 1) % len(self.opponent_attack_pattern)
            
            if action_name in self.ATTACKS:
                self.opponent_action = action_name
                self.opponent_action_timer = self.ATTACKS[action_name]["cooldown"] + 20 # Slower opponent
            else: # Block
                self.opponent_action = "block"
                self.opponent_action_timer = 40

        # -- Update Game State --
        # Momentum decay
        self.player_momentum -= self.MOMENTUM_DECAY_RATE
        reward -= 0.01

        # Clamp values
        self.player_momentum = max(0.0, min(self.PLAYER_MAX_MOMENTUM, self.player_momentum))
        self.current_opponent_health = max(0.0, self.current_opponent_health)

        # -- Calculate Rewards --
        if hit_landed: reward += 0.1
        if attack_blocked: reward -= 0.05

        if self.player_momentum >= self.PLAYER_MAX_MOMENTUM and not self.is_momentum_maxed:
            reward += 50
            self.is_momentum_maxed = True
            # // Sound effect: "POWER_UP"
        elif self.player_momentum < self.PLAYER_MAX_MOMENTUM:
            self.is_momentum_maxed = False

        # -- Check for Opponent Defeat --
        if self.current_opponent_health <= 0:
            reward += 10
            self.score += 1
            self.current_opponent_idx += 1
            # // Sound effect: "OPPONENT_DOWN"
            if self.current_opponent_idx < len(self.OPPONENT_HEALTHS):
                self.current_opponent_health = float(self.OPPONENT_HEALTHS[self.current_opponent_idx])
                self.opponent_action = "idle"
                self.opponent_action_timer = 60 # Pause before next opponent fights
            else:
                self.game_over = True # Win condition

        # -- Check Termination Conditions --
        terminated = False
        truncated = False
        if self.game_over: # Win
            reward += 100
            terminated = True
            # // Sound effect: "WIN_GAME"
        elif self.player_momentum <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
            # // Sound effect: "LOSE_GAME"
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_effects(self):
        # Update particles
        particles_to_keep = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.2 # Gravity
            p['life'] -= 1
            if p['life'] > 0:
                particles_to_keep.append(p)
        self.particles = particles_to_keep

        # Update flashes
        flashes_to_keep = []
        for f in self.hit_flashes:
            f['life'] -= 1
            if f['life'] > 0:
                flashes_to_keep.append(f)
        self.hit_flashes = flashes_to_keep

    def _create_particles(self, pos, color, count=20):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(15, 30),
                'color': color
            })
    
    def _create_hit_flash(self, pos, color, max_radius):
        self.hit_flashes.append({
            'pos': list(pos),
            'life': 8,
            'max_life': 8,
            'color': color,
            'max_radius': max_radius
        })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw ring
        ring_rect = pygame.Rect(40, self.HEIGHT - 80, self.WIDTH - 80, 40)
        pygame.draw.rect(self.screen, self.COLOR_RING, ring_rect, 4, border_radius=5)
        
        # Draw effects
        for f in self.hit_flashes:
            progress = f['life'] / f['max_life']
            radius = int(f['max_radius'] * (1.0 - progress))
            alpha = int(255 * progress)
            # Using a temp surface for alpha blending
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*f['color'], alpha), (radius, radius), radius)
            self.screen.blit(temp_surf, (int(f['pos'][0] - radius), int(f['pos'][1] - radius)))

        for p in self.particles:
            size = int(p['life'] / 6)
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), size)

        # Draw robots
        self._draw_robot(self.PLAYER_POS, self.COLOR_PLAYER, self.player_action, self.player_action_timer)
        self._draw_robot(self.OPPONENT_POS, self.COLOR_OPPONENT, self.opponent_action, self.opponent_action_timer)

    def _draw_robot(self, pos, color, action, timer):
        x, y = pos
        bob = math.sin(self.steps * 0.1) * 3 if action == "idle" else 0
        
        # Body
        torso_rect = pygame.Rect(x - 20, y - 50 + bob, 40, 60)
        pygame.draw.rect(self.screen, color, torso_rect, border_radius=8)
        
        # Head
        head_pos = (int(x), int(y - 65 + bob))
        pygame.draw.circle(self.screen, color, head_pos, 20)
        pygame.gfxdraw.aacircle(self.screen, head_pos[0], head_pos[1], 20, color)
        
        # Arms
        arm_l_pos, arm_r_pos = (x - 25, y - 30 + bob), (x + 25, y - 30 + bob)
        
        cooldown = 1
        if action in self.ATTACKS:
            cooldown = self.ATTACKS[action]["cooldown"]

        progress = max(0, timer / cooldown) if cooldown > 1 else 0

        # Animate arms based on action
        if "jab" in action:
            punch_extend = (1 - abs(progress - 0.5) * 2) * 50
            arm_l_pos = (arm_l_pos[0] - punch_extend, arm_l_pos[1])
        elif "hook" in action:
            punch_extend = (1 - abs(progress - 0.5) * 2) * 30
            arm_r_pos = (arm_r_pos[0] + punch_extend, arm_r_pos[1] - 10)
        elif "block" in action:
            arm_l_pos = (x - 10, y - 50 + bob)
            arm_r_pos = (x + 10, y - 50 + bob)

        # Draw arms (gloves)
        pygame.draw.circle(self.screen, self.COLOR_ATTACK, (int(arm_l_pos[0]), int(arm_l_pos[1])), 12)
        pygame.draw.circle(self.screen, self.COLOR_ATTACK, (int(arm_r_pos[0]), int(arm_r_pos[1])), 12)

    def _render_ui(self):
        # Player UI
        self._draw_bar(20, 20, 200, 20, self.player_momentum, self.PLAYER_MAX_MOMENTUM, self.COLOR_MOMENTUM, "MOMENTUM")
        
        # Opponent UI
        opponent_max_health = self.OPPONENT_HEALTHS[min(self.current_opponent_idx, len(self.OPPONENT_HEALTHS)-1)]
        self._draw_bar(self.WIDTH - 220, 20, 200, 20, self.current_opponent_health, opponent_max_health, self.COLOR_OPPONENT, f"OPPONENT {self.current_opponent_idx + 1}/3")

        # Game Over Text
        if self.game_over:
            if self.current_opponent_idx >= len(self.OPPONENT_HEALTHS):
                text = "YOU WIN"
                color = self.COLOR_PLAYER
            else:
                text = "MOMENTUM DEPLETED"
                color = self.COLOR_OPPONENT
            
            text_surf = self.font_large.render(text, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 40))
            self.screen.blit(text_surf, text_rect)

    def _draw_bar(self, x, y, w, h, value, max_value, color, label):
        # Background
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (x, y, w, h), border_radius=4)
        # Fill
        fill_w = (value / max_value) * w if max_value > 0 else 0
        pygame.draw.rect(self.screen, color, (x, y, int(fill_w), h), border_radius=4)
        # Border
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (x, y, w, h), 2, border_radius=4)
        # Label
        label_surf = self.font_small.render(label, True, self.COLOR_TEXT)
        self.screen.blit(label_surf, (x, y + h + 5))
        # Value Text
        val_text = f"{int(value)}/{int(max_value)}"
        val_surf = self.font_small.render(val_text, True, self.COLOR_TEXT)
        val_rect = val_surf.get_rect(center=(x + w / 2, y + h / 2))
        self.screen.blit(val_surf, val_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_momentum": self.player_momentum,
            "opponent_health": self.current_opponent_health,
            "opponents_defeated": self.current_opponent_idx
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and is not used by the evaluation system.
    # It is okay to remove it or modify it.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Momentum Boxer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default action: no-op (block)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- ENV RESET ---")

        # Map keys to actions
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1 # Left Hook
        elif keys[pygame.K_DOWN]: action[0] = 2 # Right Hook
        elif keys[pygame.K_LEFT]: action[0] = 3 # Left Jab
        elif keys[pygame.K_RIGHT]: action[0] = 4 # Right Jab
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")

        # Render to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()