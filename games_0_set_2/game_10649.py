import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:46:09.531837
# Source Brief: brief_00649.md
# Brief Index: 649
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a neon-themed 2D fighting game.
    The agent controls a character and must defeat an AI opponent.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "A neon-themed 2D fighting game where you must defeat an AI opponent "
        "using a combination of attacks and blocks."
    )
    user_guide = (
        "Use ←→ to move and ↓ to block. Press space for a light attack and shift for a heavy attack."
    )
    auto_advance = True


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.render_mode = render_mode
        pygame.init()
        pygame.font.init()
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_dmg = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)

        # --- Game Constants ---
        self.ARENA_FLOOR = 350
        self.ARENA_MARGIN = 40
        self.GRAVITY = 0.6
        self.PLAYER_SPEED = 4.0
        self.MAX_EPISODE_STEPS = 1200 # Increased from 1000 to allow for longer fights

        # --- Color Palette ---
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_ARENA_GLOW = (40, 20, 70)
        self.COLOR_ARENA_LINE = (80, 60, 120)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_PLAYER_GLOW = (0, 100, 150)
        self.COLOR_OPPONENT = (255, 50, 50)
        self.COLOR_OPPONENT_GLOW = (150, 25, 25)
        self.COLOR_BLOCK_SHIELD = (100, 150, 255, 120)
        self.COLOR_HIT_SPARK = (255, 255, 150)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_GOOD = (80, 220, 80)
        self.COLOR_HEALTH_BAD = (220, 80, 80)
        self.COLOR_HEALTH_BG = (50, 50, 50)
        self.COLOR_DMG_TEXT_PLAYER = (50, 180, 255)
        self.COLOR_DMG_TEXT_OPPONENT = (255, 180, 50)
        
        # --- Persistent State (survives resets) ---
        self.total_victories = 0

        # --- Initialize State Variables ---
        # self.reset() is called by the environment wrapper
        
        # --- Critical Self-Check ---
        # self.validate_implementation() # This can be removed for production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.game_over_timer = 120 # Frames to show message

        # Player state
        self.player_pos = np.array([160.0, self.ARENA_FLOOR])
        self.player_vel = np.array([0.0, 0.0])
        self.player_max_health = 100
        self.player_health = self.player_max_health
        self.player_action = "idle"  # idle, light_attack, heavy_attack, block, hit
        self.player_action_timer = 0
        self.player_cooldowns = {"light": 0, "heavy": 0}
        self.player_facing_right = True
        self.prev_space_held = False
        self.prev_shift_held = False

        # Opponent state
        self.opponent_pos = np.array([480.0, self.ARENA_FLOOR])
        self.opponent_vel = np.array([0.0, 0.0])
        self.opponent_max_health = int(50 * (1.05 ** self.total_victories))
        self.opponent_health = self.opponent_max_health
        self.opponent_damage_multiplier = 1.02 ** self.total_victories
        self.opponent_action = "idle"
        self.opponent_action_timer = 0
        self.opponent_cooldowns = {"light": 0, "heavy": 0}
        self.opponent_ai_state = "passive"  # passive, aggressive, defensive
        self.opponent_ai_timer = self.np_random.integers(30, 60)
        self.opponent_facing_right = False

        # Effects lists
        self.particles = []
        self.damage_texts = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            self.game_over_timer -= 1
            if self.game_over_timer <= 0:
                # This part is not strictly necessary for typical RL loops,
                # but good for interactive play. The terminated flag handles the episode end.
                pass 
            # The agent can't act when the game is over, but we still return a valid tuple
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        step_reward = 0

        # --- Update Timers and Cooldowns ---
        self._update_timers()

        # --- Handle Player Input and Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_player_actions(movement, space_held, shift_held)
        
        # --- Handle Opponent AI ---
        self._handle_opponent_ai()

        # --- Update Physics and Positions ---
        self._update_positions()

        # --- Handle Collisions and Damage ---
        step_reward += self._handle_collisions_and_damage()
        
        self.score += step_reward

        # --- Check for Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_EPISODE_STEPS
        if terminated or truncated:
            self.game_over = True
            if self.opponent_health <= 0:
                step_reward += 105 # +100 for win, +5 for event
                self.score += 105
                self.total_victories += 1
                self.game_over_message = "YOU WIN"
            elif self.player_health <= 0:
                # No explicit negative terminal reward in brief, but good practice
                # self.score -= 100 
                self.game_over_message = "YOU LOSE"
            else: # Max steps reached
                self.game_over_message = "TIME UP"


        # Update previous button states for rising edge detection
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return (
            self._get_observation(),
            step_reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_timers(self):
        if self.player_action_timer > 0: self.player_action_timer -= 1
        if self.opponent_action_timer > 0: self.opponent_action_timer -= 1
        if self.opponent_ai_timer > 0: self.opponent_ai_timer -= 1
        for cd in self.player_cooldowns: self.player_cooldowns[cd] = max(0, self.player_cooldowns[cd] - 1)
        for cd in self.opponent_cooldowns: self.opponent_cooldowns[cd] = max(0, self.opponent_cooldowns[cd] - 1)

    def _handle_player_actions(self, movement, space_held, shift_held):
        is_acting = self.player_action_timer > 0

        # Action selection (prioritize attacks)
        light_attack_triggered = space_held and not self.prev_space_held
        heavy_attack_triggered = shift_held and not self.prev_shift_held

        if not is_acting:
            # Attacks
            if heavy_attack_triggered and self.player_cooldowns["heavy"] == 0:
                self.player_action = "heavy_attack"
                self.player_action_timer = 35 # long wind-up and effect
                self.player_cooldowns["heavy"] = 70
                # SFX: Heavy Swing
            elif light_attack_triggered and self.player_cooldowns["light"] == 0:
                self.player_action = "light_attack"
                self.player_action_timer = 20 # quick
                self.player_cooldowns["light"] = 30
                # SFX: Light Swing
            # Movement and Blocking
            elif movement in [3, 4]: # Left/Right
                self.player_action = "move"
                self.player_vel[0] = self.PLAYER_SPEED if movement == 4 else -self.PLAYER_SPEED
                self.player_facing_right = (movement == 4)
            else: # None, Up, Down
                # In action space, 0 is Down, 1 is Up, 2 is None (or other unused)
                # The user guide simplifies this: Down to block.
                self.player_action = "block" if movement == 2 else "idle"
                self.player_vel[0] = 0
        else:
            self.player_vel[0] = 0 # Cannot move while in an action animation

    def _handle_opponent_ai(self):
        is_acting = self.opponent_action_timer > 0
        dist_to_player = abs(self.player_pos[0] - self.opponent_pos[0])
        
        # Update facing direction
        self.opponent_facing_right = self.player_pos[0] > self.opponent_pos[0]

        # AI state transitions
        if self.opponent_ai_timer <= 0:
            self.opponent_ai_timer = self.np_random.integers(45, 90)
            rand_val = self.np_random.random()
            if self.opponent_health < self.opponent_max_health * 0.4:
                # Desperate state: more defensive
                if rand_val < 0.6: self.opponent_ai_state = "defensive"
                else: self.opponent_ai_state = "aggressive"
            else:
                if rand_val < 0.33: self.opponent_ai_state = "passive"
                elif rand_val < 0.66: self.opponent_ai_state = "aggressive"
                else: self.opponent_ai_state = "defensive"

        # Execute action based on state
        if not is_acting:
            # Aggressive: Get close and attack
            if self.opponent_ai_state == "aggressive":
                if dist_to_player > 120:
                    self.opponent_vel[0] = self.PLAYER_SPEED * (1 if self.opponent_facing_right else -1)
                    self.opponent_action = "move"
                elif dist_to_player > 60 and self.np_random.random() < 0.05 and self.opponent_cooldowns["heavy"] == 0:
                    self.opponent_action = "heavy_attack"
                    self.opponent_action_timer = 35
                    self.opponent_cooldowns["heavy"] = 90
                elif self.np_random.random() < 0.1 and self.opponent_cooldowns["light"] == 0:
                    self.opponent_action = "light_attack"
                    self.opponent_action_timer = 20
                    self.opponent_cooldowns["light"] = 45
                else:
                    self.opponent_vel[0] = 0
                    self.opponent_action = "idle"
            # Defensive: Keep distance or block
            elif self.opponent_ai_state == "defensive":
                if dist_to_player < 100:
                    self.opponent_vel[0] = self.PLAYER_SPEED * (-1 if self.opponent_facing_right else 1)
                    self.opponent_action = "move"
                elif self.player_action in ["light_attack", "heavy_attack"]:
                     self.opponent_action = "block"
                else:
                    self.opponent_action = "idle"
            # Passive: Stand still
            else:
                self.opponent_vel[0] = 0
                self.opponent_action = "idle"
        else:
            self.opponent_vel[0] = 0


    def _update_positions(self):
        # Apply velocity
        self.player_pos += self.player_vel
        self.opponent_pos += self.opponent_vel

        # Apply gravity (if not on floor)
        if self.player_pos[1] < self.ARENA_FLOOR: self.player_vel[1] += self.GRAVITY
        if self.opponent_pos[1] < self.ARENA_FLOOR: self.opponent_vel[1] += self.GRAVITY
        
        # Floor collision
        self.player_pos[1] = min(self.player_pos[1], self.ARENA_FLOOR)
        self.opponent_pos[1] = min(self.opponent_pos[1], self.ARENA_FLOOR)

        # Arena boundary collision
        self.player_pos[0] = np.clip(self.player_pos[0], self.ARENA_MARGIN, self.screen_width - self.ARENA_MARGIN)
        self.opponent_pos[0] = np.clip(self.opponent_pos[0], self.ARENA_MARGIN, self.screen_width - self.ARENA_MARGIN)

    def _get_hitbox(self, char_pos, char_facing_right, action, action_timer):
        body = pygame.Rect(char_pos[0] - 20, char_pos[1] - 80, 40, 80)
        
        attack_phase = 0
        if action == "light_attack":
            # Attack active in the middle of the animation
            if 5 < action_timer < 15:
                attack_phase = 1
        elif action == "heavy_attack":
            # Attack active later and for longer
            if 10 < action_timer < 30:
                attack_phase = 1

        if attack_phase > 0:
            offset = 25 if char_facing_right else -65
            width = 40
            if action == "heavy_attack":
                offset = 30 if char_facing_right else -80
                width = 50
            weapon = pygame.Rect(char_pos[0] + offset, char_pos[1] - 60, width, 20)
            return body, weapon
        
        return body, None


    def _handle_collisions_and_damage(self):
        reward = 0
        
        player_body, player_weapon = self._get_hitbox(self.player_pos, self.player_facing_right, self.player_action, self.player_action_timer)
        opp_body, opp_weapon = self._get_hitbox(self.opponent_pos, self.opponent_facing_right, self.opponent_action, self.opponent_action_timer)

        # Player attacks opponent
        if player_weapon and player_weapon.colliderect(opp_body):
            if self.opponent_action == "block":
                reward += 1.0 # Successful block reward
                self._create_particles(opp_body.center, 10, self.COLOR_BLOCK_SHIELD, 2, 4)
                # SFX: Block
                self.player_action_timer = 1 # End attack on block
            else:
                damage = 15 if self.player_action == "heavy_attack" else 5
                self.opponent_health = max(0, self.opponent_health - damage)
                reward += 0.1 # Damage dealing reward
                self._create_particles(player_weapon.center, 20, self.COLOR_HIT_SPARK, 3, 6)
                self.damage_texts.append({"pos": list(opp_body.center), "text": str(damage), "timer": 40, "color": self.COLOR_DMG_TEXT_OPPONENT})
                # SFX: Hit
                self.player_action_timer = 1 # End attack on hit
                self.opponent_action = "hit"
                self.opponent_action_timer = 10
        
        # Opponent attacks player
        if opp_weapon and opp_weapon.colliderect(player_body):
            if self.player_action == "block":
                # No reward for player being blocked
                self._create_particles(player_body.center, 10, self.COLOR_BLOCK_SHIELD, 2, 4)
                # SFX: Block
                self.opponent_action_timer = 1
            else:
                damage = int((10 if self.opponent_action == "heavy_attack" else 2) * self.opponent_damage_multiplier)
                self.player_health = max(0, self.player_health - damage)
                reward -= 0.1 # Taking damage penalty
                self._create_particles(opp_weapon.center, 20, self.COLOR_HIT_SPARK, 3, 6)
                self.damage_texts.append({"pos": list(player_body.center), "text": str(damage), "timer": 40, "color": self.COLOR_DMG_TEXT_PLAYER})
                # SFX: Hit
                self.opponent_action_timer = 1
                self.player_action = "hit"
                self.player_action_timer = 10

        return reward

    def _check_termination(self):
        return self.player_health <= 0 or self.opponent_health <= 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "opponent_health": self.opponent_health,
            "victories": self.total_victories
        }

    def _render_game(self):
        # Arena floor
        floor_rect = pygame.Rect(0, self.ARENA_FLOOR, self.screen_width, self.screen_height - self.ARENA_FLOOR)
        pygame.draw.rect(self.screen, self.COLOR_ARENA_GLOW, floor_rect)
        pygame.draw.line(self.screen, self.COLOR_ARENA_LINE, (0, self.ARENA_FLOOR), (self.screen_width, self.ARENA_FLOOR), 3)

        # Update and render effects
        self._update_and_render_particles()
        self._update_and_render_damage_texts()

        # Render characters
        self._render_character("player")
        self._render_character("opponent")
        
        # Render game over message
        if self.game_over:
            s = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            text_surf = self.font_game_over.render(self.game_over_message, True, self.COLOR_UI_TEXT)
            text_rect = text_surf.get_rect(center=(self.screen_width/2, self.screen_height/2))
            self.screen.blit(text_surf, text_rect)


    def _render_character(self, who):
        if who == "player":
            pos, vel, action, timer, facing_right = self.player_pos, self.player_vel, self.player_action, self.player_action_timer, self.player_facing_right
            color, glow_color = self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW
        else:
            pos, vel, action, timer, facing_right = self.opponent_pos, self.opponent_vel, self.opponent_action, self.opponent_action_timer, self.opponent_facing_right
            color, glow_color = self.COLOR_OPPONENT, self.COLOR_OPPONENT_GLOW
        
        x, y = int(pos[0]), int(pos[1])
        
        # Hit stun wobble
        if action == "hit":
            x += self.np_random.integers(-3, 4)
            y += self.np_random.integers(-1, 2)

        # Body
        body_rect = pygame.Rect(x - 15, y - 80, 30, 80)
        glow_rect = body_rect.inflate(20, 10)
        
        # Draw glow
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(s, (*glow_color, 80), s.get_rect())
        self.screen.blit(s, glow_rect.topleft)

        # Draw body
        pygame.draw.rect(self.screen, color, body_rect, border_radius=5)
        # Head
        pygame.gfxdraw.filled_circle(self.screen, x, y - 90, 12, color)

        # Action animations
        if timer > 0:
            if action == "block":
                shield_rect = pygame.Rect(0, 0, 60, 90)
                shield_rect.center = body_rect.center
                s = pygame.Surface(shield_rect.size, pygame.SRCALPHA)
                pygame.draw.ellipse(s, self.COLOR_BLOCK_SHIELD, s.get_rect())
                self.screen.blit(s, shield_rect.topleft)
            
            elif action in ["light_attack", "heavy_attack"]:
                progress = 1.0 - (timer / (20.0 if action == "light_attack" else 35.0))
                # Simple swing arc
                arc_progress = math.sin(progress * math.pi)
                
                weapon_len = 40 if action == "light_attack" else 50
                weapon_w = 8 if action == "light_attack" else 12

                angle = (90 * arc_progress) - 45
                if not facing_right: angle = 180 - angle

                hand_x = x + (15 if facing_right else -15)
                hand_y = y - 50
                
                end_x = hand_x + weapon_len * math.cos(math.radians(angle))
                end_y = hand_y - weapon_len * math.sin(math.radians(angle))

                pygame.draw.line(self.screen, color, (hand_x, hand_y), (end_x, end_y), weapon_w)
                pygame.draw.line(self.screen, (255,255,255), (hand_x, hand_y), (end_x, end_y), weapon_w // 2)

    def _render_ui(self):
        # Player Health Bar
        self._draw_health_bar(20, 20, 200, 20, self.player_health, self.player_max_health, "Player")
        # Opponent Health Bar
        self._draw_health_bar(self.screen_width - 220, 20, 200, 20, self.opponent_health, self.opponent_max_health, "Opponent")
        
        # Score and Victories
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.screen_width / 2 - score_text.get_width()/2, 20))
        vic_text = self.font_ui.render(f"Victories: {self.total_victories}", True, self.COLOR_UI_TEXT)
        self.screen.blit(vic_text, (self.screen_width / 2 - vic_text.get_width()/2, 45))

    def _draw_health_bar(self, x, y, w, h, current, maximum, label):
        # Background
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (x, y, w, h), border_radius=3)
        # Health
        health_ratio = max(0, current / maximum)
        health_color = self.COLOR_HEALTH_GOOD if health_ratio > 0.3 else self.COLOR_HEALTH_BAD
        pygame.draw.rect(self.screen, health_color, (x, y, int(w * health_ratio), h), border_radius=3)
        # Label
        label_text = self.font_ui.render(f"{label}: {int(current)}/{int(maximum)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(label_text, (x, y + h + 5))

    def _create_particles(self, pos, count, color, min_speed, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos), "vel": vel, "timer": self.np_random.integers(20, 40), "color": color
            })

    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # gravity on particles
            p["timer"] -= 1
            if p["timer"] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p["timer"] / 40.0))
                color = (*p["color"][:3], alpha)
                radius = int(max(1, 5 * (p["timer"] / 40.0)))
                
                s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (radius, radius), radius)
                self.screen.blit(s, (int(p["pos"][0]-radius), int(p["pos"][1]-radius)))

    def _update_and_render_damage_texts(self):
        for dt in self.damage_texts[:]:
            dt["pos"][1] -= 0.5 # Float up
            dt["timer"] -= 1
            if dt["timer"] <= 0:
                self.damage_texts.remove(dt)
            else:
                alpha = int(255 * (dt["timer"] / 40.0))
                text_surf = self.font_dmg.render(dt["text"], True, dt["color"])
                text_surf.set_alpha(alpha)
                self.screen.blit(text_surf, (int(dt["pos"][0]), int(dt["pos"][1])))

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to test the environment interactively.
    # It sets up a pygame window and maps keyboard keys to actions.
    
    # Un-dummy the video driver for interactive mode
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Neon Fighter Arena")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    
    # Action mapping: 0:None, 1:Up, 2:Down, 3:Left, 4:Right
    # The user guide simplifies this for players.
    action = [0, 0, 0] 
    
    while running:
        # --- Human Controls ---
        movement = 0 # Default: idle/no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'R' key
                    print("Resetting environment...")
                    obs, info = env.reset()
                    total_reward = 0
                if event.key == pygame.K_ESCAPE:
                    running = False

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            # The env will show a game over screen. We wait for the user to press 'R' or quit.
            pass

        # --- Rendering ---
        # The observation is the rendered frame, so we just need to display it.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Run at 30 FPS

    env.close()
    pygame.quit()