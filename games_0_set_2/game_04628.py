import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper classes for game objects
class Pin:
    def __init__(self, x, y):
        self.initial_pos = pygame.Vector2(x, y)
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(0, 0)
        self.angle = 0
        self.angular_vel = 0
        self.is_standing = True
        self.radius = 8

    def reset(self):
        self.pos = pygame.Vector2(self.initial_pos)
        self.vel = pygame.Vector2(0, 0)
        self.angle = 0
        self.angular_vel = 0
        self.is_standing = True

    def update(self):
        if not self.is_standing:
            self.pos += self.vel
            self.vel *= 0.95  # Friction
            self.angle += self.angular_vel
            self.angular_vel *= 0.98

    def hit(self, impact_vel, hit_point):
        if self.is_standing:
            self.is_standing = False
            # sfx: Pin hit sound
            self.vel = pygame.Vector2(impact_vel) * 0.6
            offset = (hit_point - self.pos).normalize() if (hit_point - self.pos).length() > 0 else pygame.Vector2(1,0)
            self.angular_vel = offset.x * impact_vel.length() * 0.05 + random.uniform(-0.5, 0.5)
            return True
        return False

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to aim, ↑↓ to adjust power. Press space to roll the ball."
    )

    game_description = (
        "Top-down arcade bowling. Score 150 points to win, but don't let your score drop below 50!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.LANE_WIDTH = 200
        self.LANE_Y_START = 50
        self.LANE_Y_END = self.SCREEN_HEIGHT - 20
        self.LANE_X_CENTER = self.SCREEN_WIDTH / 2

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_LANE = (100, 80, 60)
        self.COLOR_GUTTER = (60, 50, 40)
        self.COLOR_PIN = (240, 240, 240)
        self.COLOR_PIN_FALLEN = (120, 120, 120)
        self.COLOR_BALL = (60, 150, 255)
        self.COLOR_AIM = (100, 255, 100, 150)
        self.COLOR_POWER_BG = (50, 50, 50)
        self.COLOR_POWER_FILL = (255, 50, 50)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_WIN = (100, 255, 100)
        self.COLOR_LOSE = (255, 100, 100)
        self.COLOR_SPARK = (255, 220, 100)
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_title = pygame.font.SysFont("monospace", 48, bold=True)

        # Game state variables are initialized in reset()
        self.pins = []
        self._create_pin_objects()
        
        # This check is not part of the standard API but useful for dev
        # self.validate_implementation()

    def _create_pin_objects(self):
        self.pins = []
        pin_layout = [(0, 0), (-1, 1), (1, 1), (-2, 2), (0, 2), (2, 2), (-3, 3), (-1, 3), (1, 3), (3, 3)]
        pin_spacing_x = 25
        pin_spacing_y = 20
        pin_start_y = 100
        for dx, dy in pin_layout:
            x = self.LANE_X_CENTER + dx * pin_spacing_x / 2
            y = pin_start_y + dy * pin_spacing_y
            self.pins.append(Pin(x, y))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 75
        self.game_over = False
        self.game_phase = 'aiming' # 'aiming', 'rolling'
        
        self.ball_pos = pygame.Vector2(0, 0)
        self.ball_vel = pygame.Vector2(0, 0)
        self.ball_radius = 10

        self.aim_pos_x = self.LANE_X_CENTER
        self.power_level = 0.5 # 0.0 to 1.0
        self.roll_timer = 0
        
        self.particles = []

        for pin in self.pins:
            pin.reset()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        terminated = False

        if self.game_phase == 'aiming':
            # Adjust aim
            if movement == 3: # Left
                self.aim_pos_x -= 3
            elif movement == 4: # Right
                self.aim_pos_x += 3
            self.aim_pos_x = np.clip(self.aim_pos_x, self.LANE_X_CENTER - self.LANE_WIDTH/2 + self.ball_radius, self.LANE_X_CENTER + self.LANE_WIDTH/2 - self.ball_radius)
            
            # Adjust power
            if movement == 1: # Up
                self.power_level += 0.05
            elif movement == 2: # Down
                self.power_level -= 0.05
            self.power_level = np.clip(self.power_level, 0, 1)

            # Launch ball
            if space_held:
                # sfx: Ball roll start
                self.game_phase = 'rolling'
                self.ball_pos = pygame.Vector2(self.aim_pos_x, self.LANE_Y_END - 30)
                max_speed = 15
                self.ball_vel = pygame.Vector2(0, -max_speed * self.power_level)
                self.roll_timer = 150 # Max steps for a roll

        elif self.game_phase == 'rolling':
            self._update_physics()
            self.roll_timer -= 1
            
            is_roll_over = self.ball_vel.length() < 0.1 or self.roll_timer <= 0
            
            if is_roll_over:
                pins_down_this_turn = sum(1 for pin in self.pins if not pin.is_standing)
                turn_score_change = pins_down_this_turn
                reward = float(pins_down_this_turn)

                # Gutter ball check
                is_gutter = pins_down_this_turn == 0 and self.ball_pos.y < self.LANE_Y_START + 50
                if is_gutter:
                    turn_score_change -= 2
                    reward -= 2.0
                
                # Strike check
                if pins_down_this_turn == 10:
                    turn_score_change += 5
                    reward += 5.0
                    # sfx: Strike!
                
                self.score += turn_score_change
                
                # Check for game termination
                if self.score >= 150:
                    terminated = True
                    reward += 100.0
                    # sfx: Win jingle
                elif self.score < 50:
                    terminated = True
                    reward -= 100.0
                    # sfx: Lose sound
                
                # Reset for next turn
                self.game_phase = 'aiming'
                for pin in self.pins:
                    pin.reset()

        self.steps += 1
        if self.steps >= 1000:
            terminated = True
        
        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_physics(self):
        # Update ball
        self.ball_pos += self.ball_vel
        self.ball_vel *= 0.99 # Friction

        # Gutter check
        if self.ball_pos.x < self.LANE_X_CENTER - self.LANE_WIDTH/2 or \
           self.ball_pos.x > self.LANE_X_CENTER + self.LANE_WIDTH/2:
            self.ball_vel.x *= -0.5 # Bounce off gutter walls slightly
            if self.ball_pos.y < self.LANE_Y_START + 50: # If in gutter past pins
                self.ball_vel *= 0 # Stop ball in gutter
        
        # Ball-pin collisions
        for pin in self.pins:
            if pin.is_standing:
                dist_vec = self.ball_pos - pin.pos
                if dist_vec.length() < self.ball_radius + pin.radius:
                    if pin.hit(self.ball_vel, self.ball_pos):
                        self._create_sparks(pin.pos)
                        # Apply impulse
                        impulse = dist_vec.normalize() * self.ball_vel.length() * 0.4 if dist_vec.length() > 0 else pygame.Vector2()
                        self.ball_vel -= impulse
                        pin.vel += impulse
        
        # Pin-pin collisions and pin updates
        falling_pins = [p for p in self.pins if not p.is_standing and p.vel.length() > 0.1]
        standing_pins = [p for p in self.pins if p.is_standing]
        
        for p1 in falling_pins:
            for p2 in standing_pins:
                dist_vec = p1.pos - p2.pos
                if dist_vec.length() < p1.radius + p2.radius:
                     if p2.hit(p1.vel, p1.pos):
                        self._create_sparks(p2.pos)
                        impulse = dist_vec.normalize() * p1.vel.length() * 0.8 if dist_vec.length() > 0 else pygame.Vector2()
                        p1.vel -= impulse
                        p2.vel += impulse

        for pin in self.pins:
            pin.update()

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 0.05

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        # Draw gutters
        gutter_rect = pygame.Rect(self.LANE_X_CENTER - self.LANE_WIDTH/2 - 20, 0, self.LANE_WIDTH + 40, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GUTTER, gutter_rect)
        
        # Draw lane
        lane_rect = pygame.Rect(self.LANE_X_CENTER - self.LANE_WIDTH/2, 0, self.LANE_WIDTH, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_LANE, lane_rect)

        # Draw pins
        for pin in self.pins:
            color = self.COLOR_PIN if pin.is_standing else self.COLOR_PIN_FALLEN
            if pin.is_standing:
                pygame.gfxdraw.filled_circle(self.screen, int(pin.pos.x), int(pin.pos.y), pin.radius, color)
                pygame.gfxdraw.aacircle(self.screen, int(pin.pos.x), int(pin.pos.y), pin.radius, color)
            else:
                # Draw fallen pin as a rotated capsule
                p1 = pin.pos + pygame.Vector2(0, -pin.radius).rotate(-pin.angle)
                p2 = pin.pos + pygame.Vector2(0, pin.radius).rotate(-pin.angle)
                pygame.draw.line(self.screen, color, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), pin.radius)


        # Draw ball
        if self.game_phase == 'rolling':
            pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.ball_radius, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.ball_radius, self.COLOR_BALL)

        # Draw particles
        for p in self.particles:
            alpha = max(0, p['life'] * 255)
            # Pygame doesn't handle alpha well in this context, so we'll just draw solid
            color = self.COLOR_SPARK 
            size = int(p['life'] * 4)
            if size > 0:
                pygame.draw.circle(self.screen, color, (int(p['pos'].x), int(p['pos'].y)), size)


    def _render_ui(self):
        # Draw Score
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Draw Aiming UI
        if self.game_phase == 'aiming':
            # Power bar
            power_bar_rect_bg = pygame.Rect(10, 40, 150, 20)
            pygame.draw.rect(self.screen, self.COLOR_POWER_BG, power_bar_rect_bg)
            power_bar_width = int(146 * self.power_level)
            power_bar_rect_fill = pygame.Rect(12, 42, power_bar_width, 16)
            pygame.draw.rect(self.screen, self.COLOR_POWER_FILL, power_bar_rect_fill)

            # Aiming line
            start_pos = (int(self.aim_pos_x), self.LANE_Y_END - 30)
            end_pos = (int(self.aim_pos_x), self.LANE_Y_START)
            pygame.draw.aaline(self.screen, self.COLOR_AIM, start_pos, end_pos)
            
            # Draw aiming ball position
            pygame.gfxdraw.filled_circle(self.screen, start_pos[0], start_pos[1], self.ball_radius, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, start_pos[0], start_pos[1], self.ball_radius, self.COLOR_BALL)

        # Draw controls guide
        guide_text = self.font_small.render(self.user_guide, True, self.COLOR_TEXT)
        self.screen.blit(guide_text, (10, self.SCREEN_HEIGHT - 25))

        # Draw Game Over/Win message
        if self.game_over:
            if self.score >= 150:
                msg = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                msg = "GAME OVER"
                color = self.COLOR_LOSE
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_title.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _create_sparks(self, pos):
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': random.uniform(0.5, 1.0)
            })

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game directly
    # It requires a display, so we unset the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    print("\n" + "="*40)
    print(f"Game: {env.game_description}")
    print(f"Controls: {env.user_guide}")
    print("="*40 + "\n")

    # To display the game, we need to create a real screen
    pygame.display.set_caption("Arcade Bowling")
    real_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Map keyboard inputs to MultiDiscrete action space
        keys = pygame.key.get_pressed()
        action.fill(0) # Reset actions
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Reward: {reward:.2f}")

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit to 30 FPS for playability

    print("Game Over!")
    env.close()